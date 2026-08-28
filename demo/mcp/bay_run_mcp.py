"""
Bay Run MCP server (auth-enabled) — a compatibility connector for older stdio-only clients.

Prefer the canonical live MCP front door, whose current default tools are exactly
coprocessor, run_pin, and solve_task:
The legacy connector below exposes an older compatibility subset when run locally:
    find_specialist_for_task · request_specialist · route · discover_models · eval_models ·
    embed · rerank · classify · extract

This does not mirror the current default remote MCP inventory. Use it only when an older stdio client
needs this compatibility subset; it forwards your bearer token to the service.

Requires:  pip install "mcp[cli]" httpx
Env:       BAY_RUN_BASE_URL   (default: the live service)
           BAY_RUN_TOKEN      (optional; otherwise self-mints bounded OAuth client_credentials)
Run:       python bay_run_mcp.py         # stdio transport
"""
import os
import httpx
from mcp.server.fastmcp import FastMCP

BASE = os.environ.get("BAY_RUN_BASE_URL", "https://bay-run-mvp-889989800693.us-central1.run.app")
TOKEN = os.environ.get("BAY_RUN_TOKEN")

mcp = FastMCP("bay-run")


def _post(path: str, payload: dict, timeout: int) -> dict:
    global TOKEN
    with httpx.Client(timeout=timeout) as client:
        if not TOKEN:
            response = client.post(f"{BASE}/oauth/token", json={"grant_type": "client_credentials"})
            response.raise_for_status()
            TOKEN = response.json()["access_token"]
        headers = {"authorization": f"Bearer {TOKEN}"}
        resp = client.post(f"{BASE}{path}", headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()


@mcp.tool()
def find_specialist_for_task(task: str, my_examples: list[dict], kind: str = "any") -> dict:
    """ONE call: discover -> eval -> serve pointer. Searches the 147K-model catalog, bakes the
    top candidates off on YOUR labeled examples, and returns the WINNER model id + scorecard +
    a ready-to-call serving block. Default entry point when you HAVE labeled examples.
    kind = embedding | rerank | any. Each example: {query, positive, negatives:[...]}."""
    disc = _post("/v1/discover", {"query": task, "kind": kind, "limit": 5}, 60)
    cands = [c.get("model") for c in disc.get("candidates", [])]
    cands = [c for c in cands if c][:5]
    et = "rerank" if kind == "rerank" else "embedding"
    ev = _post("/v1/eval", {"task": et, "models": cands, "dataset": my_examples, "k": 5}, 600)
    return {"winner": ev.get("winner"), "scorecard": ev.get("scorecard"),
            "serve_endpoint": f"{BASE}/v1/{'rerank' if et == 'rerank' else 'embeddings'}"}


@mcp.tool()
def route(task_hint: str, kind: str = "auto", serve: bool = False, input=None,
          query: str | None = None, documents: list[str] | None = None,
          content: str | None = None, schema: dict | None = None) -> dict:
    """RUNTIME auto-router: hand Bay Run a job you can't name a model for and it picks the best
    mirrored specialist per-request at inference time — zero examples, instant. kind =
    embedding | rerank | generative | auto. Set serve=True + the matching inputs (embedding:
    input; rerank: query+documents; generative: content, optional schema) to route AND serve in
    one call. HEURISTIC — the pick is UNPROVEN; prove it with find_specialist_for_task."""
    payload: dict = {"task_hint": task_hint, "kind": kind}
    if serve:
        payload["serve"] = True
        for k, v in (("input", input), ("query", query), ("documents", documents),
                     ("content", content), ("schema", schema)):
            if v is not None:
                payload[k] = v
    return _post("/v1/route", payload, 600)


@mcp.tool()
def discover_models(query: str, kind: str = "any", limit: int = 10) -> dict:
    """Find candidate task-specialist models in Hugging Bay's 147K-model catalog for a task.
    kind: embedding | llm | vision | audio | tool | agent | any. Returns ranked candidates
    (mirrored-first = instant serve). Candidates, NOT proven — bake them off with eval_models."""
    return _post("/v1/discover", {"query": query, "kind": kind, "limit": limit}, 60)


@mcp.tool()
def eval_models(task: str, models: list[str], dataset: list[dict], k: int = 5) -> dict:
    """Bake off N candidate models on YOUR labeled data; returns a ranked scorecard + winner.
    task = 'embedding' | 'rerank' | 'extraction' | 'generation'. Ranking datasets:
    {query, positive, negatives:[...]}. Ranked by MRR (also hit@k, ndcg@k). First call
    cold-loads models; be patient."""
    return _post("/v1/eval", {"task": task, "models": models, "dataset": dataset, "k": k}, 600)


@mcp.tool()
def embed(model: str, input: str | list[str]) -> dict:
    """Embed text with an open embedding model (curated or any HF id, loaded on demand),
    OpenAI /v1/embeddings-compatible. Pass model="auto" to let the runtime router pick."""
    return _post("/v1/embeddings", {"model": model, "input": input}, 120)


@mcp.tool()
def rerank(model: str, query: str, documents: list[str], top_n: int | None = None) -> dict:
    """Rerank documents against a query with an open cross-encoder, loaded on demand
    (Cohere/Jina-shaped). Pass model="auto" to let the runtime router pick."""
    payload: dict = {"model": model, "query": query, "documents": documents}
    if top_n is not None:
        payload["top_n"] = top_n
    return _post("/v1/rerank", payload, 120)


@mcp.tool()
def extract(model: str, content: str, schema: dict | None = None,
            instructions: str | None = None) -> dict:
    """Turn messy HTML/text (e.g. a scraper/Firecrawl dump) into schema-guided STRUCTURED JSON
    using a small CPU-served generative specialist. Returns {data, json_valid, raw} —
    BEST-EFFORT: always CHECK json_valid before trusting data. Deterministic (greedy)."""
    if schema is not None:
        response_format = {"type": "json_schema", "json_schema": {"schema": schema}}
    else:
        response_format = {"type": "json_object"}
    msg = (instructions + "\n\n" if instructions else "") + content
    resp = _post("/v1/chat/completions", {
        "model": model, "messages": [{"role": "user", "content": msg}],
        "response_format": response_format, "temperature": 0.0,
    }, 600)
    return {"model": model, "data": resp.get("parsed"), "json_valid": resp.get("json_valid"),
            "raw": resp["choices"][0]["message"]["content"], "usage": resp.get("usage")}


@mcp.tool()
def classify(model: str, input: str | list[str], candidate_labels: list[str] | None = None,
             multi_label: bool = False, top_k: int | None = None) -> dict:
    """Classify text with a small CPU-served specialist — guardrail / moderation / sentiment /
    intent / NLI. FIXED-LABEL: model = any HF sequence-classification id (e.g.
    'protectai/deberta-v3-base-prompt-injection-v2', 'unitary/toxic-bert') -> the model's own
    labels + scores. ZERO-SHOT: pass candidate_labels with an NLI model like
    'facebook/bart-large-mnli' (or model='auto') -> entailment-scored labels, no training.
    Returns {model, labels:[{label,score}], zero_shot}."""
    payload: dict = {"model": model, "input": input, "multi_label": multi_label}
    if candidate_labels is not None:
        payload["candidate_labels"] = candidate_labels
    if top_k is not None:
        payload["top_k"] = top_k
    return _post("/v1/classify", payload, 300)


@mcp.tool()
def request_specialist(task: str, examples: list[dict] | None = None, kind: str = "any") -> dict:
    """Ask Bay Run for a specialist — and NEVER get a dead end. If a servable specialist exists,
    chains discover -> (eval if you pass examples) -> a ready-to-call serve pointer. If none
    exists yet, RECORDS your demand as a pull signal and returns {status:'recorded',
    subscribe_hint}. kind = embedding|rerank|generative|classification|zeroshot|any."""
    payload: dict = {"task": task, "kind": kind}
    if examples is not None:
        payload["examples"] = examples
    return _post("/v1/request_specialist", payload, 600)


if __name__ == "__main__":
    mcp.run()
