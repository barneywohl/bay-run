"""
Bay Run MCP server (auth-enabled) — point any MCP agent at the LIVE Bay Run service.

Exposes four tools so an agent can run the whole loop autonomously:
    discover_models · eval_models · embed · rerank

This is the shipped stub (../../2026-08-07/run-mvp/mcp/mcp_server.py) plus the one
thing it needs to talk to the deployed service: it forwards your bearer token.

Requires:  pip install "mcp[cli]" httpx
Env:       BAY_RUN_BASE_URL   (default: the live service)
           BAY_RUN_TOKEN      (required for the live service)
Run:       python bay_run_mcp.py         # stdio transport
"""
import os
import httpx
from mcp.server.fastmcp import FastMCP

BASE = os.environ.get("BAY_RUN_BASE_URL", "https://bay-run-mvp-zfmlsu2yla-uc.a.run.app")
TOKEN = os.environ.get("BAY_RUN_TOKEN", "")
HEADERS = {"authorization": f"Bearer {TOKEN}"} if TOKEN else {}

mcp = FastMCP("bay-run")


def _post(path: str, payload: dict, timeout: int) -> dict:
    with httpx.Client(timeout=timeout, headers=HEADERS) as client:
        resp = client.post(f"{BASE}{path}", json=payload)
        resp.raise_for_status()
        return resp.json()


@mcp.tool()
def discover_models(query: str, kind: str = "any", limit: int = 10) -> dict:
    """Find candidate task-specialist models in Hugging Bay's catalog for a task.
    kind: embedding | rerank | any. Returns ranked candidates (mirrored-first =
    instant serve). Candidates, NOT proven-good — bake them off with eval_models next."""
    return _post("/v1/discover", {"query": query, "kind": kind, "limit": limit}, 60)


@mcp.tool()
def eval_models(task: str, models: list[str], dataset: list[dict], k: int = 5) -> dict:
    """Bake off N candidate models on YOUR labeled data; returns a ranked scorecard + winner.
    task = 'embedding' | 'rerank'. Each dataset item: {query, positive, negatives:[...]}.
    Ranked by MRR (also hit@k, ndcg@k). First call cold-loads models; be patient."""
    return _post("/v1/eval", {"task": task, "models": models, "dataset": dataset, "k": k}, 600)


@mcp.tool()
def embed(model: str, input: str | list[str]) -> dict:
    """Embed text with an open embedding model (curated or any HF id, loaded on demand)."""
    return _post("/v1/embeddings", {"model": model, "input": input}, 120)


@mcp.tool()
def rerank(model: str, query: str, documents: list[str], top_n: int | None = None) -> dict:
    """Rerank documents against a query with an open cross-encoder, loaded on demand."""
    payload: dict = {"model": model, "query": query, "documents": documents}
    if top_n is not None:
        payload["top_n"] = top_n
    return _post("/v1/rerank", payload, 120)


if __name__ == "__main__":
    mcp.run()
