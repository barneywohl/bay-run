"""
Bay Run — LlamaIndex tool wrappers.

Copy-paste to give a LlamaIndex agent the discover→eval→serve loop for open
specialist models. Points at the LIVE Bay Run service. FunctionTool derives the
tool description from each function's docstring, so they're written for selection.

    pip install llama-index-core httpx
    export BAY_RUN_TOKEN=...   # get one at https://huggingbay.xyz
"""
import os
import httpx
from llama_index.core.tools import FunctionTool

BASE = os.environ.get("BAY_RUN_BASE_URL", "https://bay-run-mvp-zfmlsu2yla-uc.a.run.app")
HEADERS = {"Authorization": f"Bearer {os.environ.get('BAY_RUN_TOKEN', '')}",
           "Content-Type": "application/json"}


def _post(path: str, payload: dict, timeout: int = 120) -> dict:
    with httpx.Client(timeout=timeout) as c:
        r = c.post(f"{BASE}{path}", headers=HEADERS, json=payload)
        r.raise_for_status()
        return r.json()


def find_specialist_for_task(task: str, my_examples: list, kind: str = "any") -> dict:
    """Find the best small open specialist model for a narrow, high-volume task
    (embeddings, reranking, classification, extraction, routing) and prove it on YOUR
    examples — cheaper per-call than a frontier API. task = plain-language description;
    my_examples = labeled items each {query, positive, negatives:[...]}; kind =
    embedding|rerank|any. Returns winning model id + scorecard + serving endpoint.
    Use instead of guessing a model name."""
    disc = _post("/v1/discover", {"query": task, "kind": kind, "limit": 5}, timeout=60)
    # discover.py contract: candidates[].model  (NOT results[].id)
    cands = [c.get("model") for c in disc.get("candidates", [])]
    cands = [c for c in cands if c][:5]
    eval_task = "rerank" if kind == "rerank" else "embedding"
    ev = _post("/v1/eval", {"task": eval_task, "models": cands,
                            "dataset": my_examples, "k": 5}, timeout=600)
    return {"winner": ev.get("winner"), "scorecard": ev.get("scorecard"),
            "serve_endpoint": f"{BASE}/v1/{'rerank' if eval_task=='rerank' else 'embeddings'}"}


def discover_models(query: str, kind: str = "any", limit: int = 5) -> dict:
    """Search a 147K-model catalog for candidate open specialist models for a task
    (embedding|llm|vision|audio|tool|agent|any). Returns ranked candidate model ids,
    mirrored-first (instant serve). Candidates only — pass to eval_models."""
    return _post("/v1/discover", {"query": query, "kind": kind, "limit": limit}, timeout=60)


def eval_models(task: str, models: list, dataset: list) -> dict:
    """Bake off candidate models on YOUR labeled data; return ranked scorecard + winner
    (leaderboard rank does not predict your-domain fit). task = embedding|rerank;
    dataset items = {query, positive, negatives:[...]}."""
    return _post("/v1/eval", {"task": task, "models": models, "dataset": dataset, "k": 5}, timeout=600)


def embed(model: str, input) -> dict:
    """Embed text with any open embedding model, served instantly (OpenAI-compatible).
    model = any HF embedding id; input = string or list of strings."""
    return _post("/v1/embeddings", {"model": model, "input": input})


def rerank(model: str, query: str, documents: list, top_n: int = None) -> dict:
    """Rerank documents by relevance to a query using any open cross-encoder — sharpen
    RAG/search precision. model = any HF reranker id; optional top_n."""
    payload = {"model": model, "query": query, "documents": documents}
    if top_n is not None:
        payload["top_n"] = top_n
    return _post("/v1/rerank", payload)


def extract(model: str, content: str, schema: dict = None, instructions: str = None) -> dict:
    """Turn messy HTML/text (e.g. a scraper/Firecrawl dump) into schema-guided STRUCTURED
    JSON using a small CPU-served generative specialist — the reliable extraction layer
    after scrapers, cheap at volume. model = a generative HF id (or the winner from
    find_specialist_for_task/eval_models); content = raw HTML/text; schema = an OpenAI-style
    JSON schema {properties, required} pinning the fields; optional instructions. Returns
    {data, json_valid, raw, usage} — BEST-EFFORT: always CHECK json_valid before trusting
    data. Deterministic (greedy)."""
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


BAY_RUN_TOOLS = [FunctionTool.from_defaults(fn=f) for f in
                 (find_specialist_for_task, discover_models, eval_models, embed, rerank, extract)]
# Usage: agent = FunctionAgent(tools=BAY_RUN_TOOLS, llm=llm)
#
# Remote MCP alternative (no wrappers needed): point an MCP client at
# https://bay-run-mvp-zfmlsu2yla-uc.a.run.app/mcp/ with an Authorization: Bearer header.
