"""
Bay Run — LangChain tool wrappers.

Copy-paste into a LangChain agent so it can discover→eval→serve open specialist
models autonomously. Points at the LIVE Bay Run service. Docstrings are written for
LangChain's tool-selection (the agent reads the @tool docstring).

    pip install langchain-core httpx
    # Runs copy-paste with self-minted OAuth client_credentials.
"""
import os
import httpx
from langchain_core.tools import tool

BASE = os.environ.get("BAY_RUN_BASE_URL", "https://bay-run-mvp-889989800693.us-central1.run.app")
TOKEN = os.environ.get("BAY_RUN_TOKEN")


def _post(path: str, payload: dict, timeout: int = 120) -> dict:
    global TOKEN
    with httpx.Client(timeout=timeout) as c:
        if not TOKEN:
            oauth = c.post(f"{BASE}/oauth/token", json={"grant_type": "client_credentials"})
            oauth.raise_for_status()
            TOKEN = oauth.json()["access_token"]
        r = c.post(f"{BASE}{path}", headers={"Authorization": f"Bearer {TOKEN}"}, json=payload)
        r.raise_for_status()
        return r.json()


@tool
def bay_run_find_specialist(task: str, my_examples: list[dict], kind: str = "any") -> dict:
    """Find the best small open specialist model for a narrow, high-volume task
    (embeddings, reranking, classification, extraction, routing) and prove it on YOUR
    examples — cheaper per-call than a frontier API. `task` = plain-language description;
    `my_examples` = a few labeled items, each {"query","positive","negatives":[...]};
    `kind` = embedding|rerank|any. Returns the winning model id + scorecard + serving
    endpoint. Use this instead of guessing a model name."""
    disc = _post("/v1/discover", {"query": task, "kind": kind, "limit": 5}, timeout=60)
    # discover.py contract: candidates[].model  (NOT results[].id)
    cands = [c.get("model") for c in disc.get("candidates", [])]
    cands = [c for c in cands if c][:5]
    eval_task = "rerank" if kind == "rerank" else "embedding"
    ev = _post("/v1/eval", {"task": eval_task, "models": cands,
                            "dataset": my_examples, "k": 5}, timeout=600)
    return {"winner": ev.get("winner"), "scorecard": ev.get("scorecard"),
            "serve_endpoint": f"{BASE}/v1/{'rerank' if eval_task=='rerank' else 'embeddings'}"}


@tool
def bay_run_discover_models(query: str, kind: str = "any", limit: int = 5) -> dict:
    """Search a 147K-model catalog for candidate open specialist models for a task
    (embedding|llm|vision|audio|tool|agent|any). Returns ranked candidate model ids,
    mirrored-first (instant serve). Candidates only — pass them to bay_run_eval_models."""
    return _post("/v1/discover", {"query": query, "kind": kind, "limit": limit}, timeout=60)


@tool
def bay_run_eval_models(task: str, models: list[str], dataset: list[dict]) -> dict:
    """Bake off candidate models on YOUR labeled data and return a ranked scorecard +
    winner (leaderboard rank does not predict your-domain fit). `task`=embedding|rerank;
    `dataset` items = {"query","positive","negatives":[...]}."""
    return _post("/v1/eval", {"task": task, "models": models, "dataset": dataset, "k": 5}, timeout=600)


@tool
def bay_run_embed(model: str, input) -> dict:
    """Embed text with any open embedding model, served instantly (OpenAI-compatible).
    `model` = any HF embedding id; `input` = string or list of strings."""
    return _post("/v1/embeddings", {"model": model, "input": input})


@tool
def bay_run_rerank(model: str, query: str, documents: list[str], top_n: int = None) -> dict:
    """Rerank documents by relevance to a query using any open cross-encoder — sharpen
    RAG/search precision. `model` = any HF reranker id; optional `top_n`."""
    payload = {"model": model, "query": query, "documents": documents}
    if top_n is not None:
        payload["top_n"] = top_n
    return _post("/v1/rerank", payload)


@tool
def bay_run_extract(model: str, content: str, schema: dict = None, instructions: str = None) -> dict:
    """Turn messy HTML/text (e.g. a scraper/Firecrawl dump) into schema-guided STRUCTURED
    JSON using a small CPU-served generative specialist — the reliable extraction layer
    after scrapers, cheap at volume. `model` = a generative HF id (or the winner from
    bay_run_find_specialist/bay_run_eval_models); `content` = the raw HTML/text; `schema` =
    an OpenAI-style JSON schema {"properties":{...},"required":[...]} pinning the fields;
    optional `instructions`. Returns {data, json_valid, raw, usage} — this is BEST-EFFORT:
    always CHECK `json_valid` before trusting `data`. Deterministic (greedy)."""
    if schema is not None:
        response_format = {"type": "json_schema", "json_schema": {"schema": schema}}
    else:
        response_format = {"type": "json_object"}
    msg = (instructions + "\n\n" if instructions else "") + content
    resp = _post("/v1/chat/completions", {
        "model": model, "messages": [{"role": "user", "content": msg}],
        "response_format": response_format, "temperature": 0.0,
    }, timeout=600)
    return {"model": model, "data": resp.get("parsed"), "json_valid": resp.get("json_valid"),
            "raw": resp["choices"][0]["message"]["content"], "usage": resp.get("usage")}


@tool
def bay_run_route(task_hint: str, kind: str = "auto", serve: bool = False,
                  input=None, query: str = None, documents: list = None,
                  content: str = None, schema: dict = None) -> dict:
    """RUNTIME auto-router: hand Bay Run a job you can't name a model for and it picks the
    best mirrored specialist per-request at inference time — zero examples, instant. `kind` =
    embedding|rerank|generative|auto. Set `serve=True` plus the matching inputs (embedding:
    `input`; rerank: `query`+`documents`; generative: `content`, optional `schema`) to route
    AND serve in one call. Returns {routed_model, routing_reason, mirrored,
    candidates_considered, serve, note}. HEURISTIC (mirrored-first -> kind-match ->
    verified_runs/downloads) — the pick is UNPROVEN; prove it with bay_run_find_specialist.
    The same router is also reachable as model=\"auto\" on bay_run_embed/bay_run_rerank."""
    payload = {"task_hint": task_hint, "kind": kind}
    if serve:
        payload["serve"] = True
        if input is not None:
            payload["input"] = input
        if query is not None:
            payload["query"] = query
        if documents is not None:
            payload["documents"] = documents
        if content is not None:
            payload["content"] = content
        if schema is not None:
            payload["schema"] = schema
    return _post("/v1/route", payload, timeout=600)


@tool
def bay_run_classify(model: str, input, candidate_labels: list[str] = None,
                     multi_label: bool = False, top_k: int = None) -> dict:
    """Classify text with a small CPU-served specialist — the guardrail / safety / moderation /
    sentiment / intent / NLI layer agents need. FIXED-LABEL: `model` = any HF
    sequence-classification id (e.g. 'protectai/deberta-v3-base-prompt-injection-v2',
    'unitary/toxic-bert', a sentiment model) -> the model's OWN labels + scores. ZERO-SHOT: pass
    `candidate_labels` (your own labels) with an NLI model like 'facebook/bart-large-mnli' (or
    model='auto') -> entailment-scored labels, no training. `multi_label`=True scores labels
    independently; `top_k` keeps the best K. Returns {model, labels:[{label,score}], zero_shot}."""
    payload = {"model": model, "input": input, "multi_label": multi_label}
    if candidate_labels is not None:
        payload["candidate_labels"] = candidate_labels
    if top_k is not None:
        payload["top_k"] = top_k
    return _post("/v1/classify", payload, timeout=300)


@tool
def bay_run_request_specialist(task: str, examples: list[dict] = None, kind: str = "any") -> dict:
    """Ask Bay Run for a specialist — and NEVER get a dead end. If a servable specialist EXISTS,
    chains discover -> (eval, if you pass `examples`) -> returns a ready-to-call serve pointer. If
    NONE exists yet, RECORDS your demand as a pull signal and returns {status:'recorded',
    subscribe_hint}. `task` = plain-language description; optional `examples` follow the
    find_specialist shape; `kind` = embedding|rerank|generative|classification|zeroshot|any. Use
    as your safe default when unsure Bay Run already covers the task."""
    payload = {"task": task, "kind": kind}
    if examples is not None:
        payload["examples"] = examples
    return _post("/v1/request_specialist", payload, timeout=600)


BAY_RUN_TOOLS = [bay_run_find_specialist, bay_run_request_specialist, bay_run_route,
                 bay_run_discover_models, bay_run_eval_models, bay_run_embed, bay_run_rerank,
                 bay_run_classify, bay_run_extract]
# Usage: agent = create_react_agent(llm, BAY_RUN_TOOLS)
#
# Remote MCP alternative (no wrappers needed): point an MCP client at
# https://bay-run-mvp-889989800693.us-central1.run.app/mcp/ with an Authorization: Bearer header.
