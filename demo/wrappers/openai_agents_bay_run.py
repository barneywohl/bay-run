"""
Bay Run — OpenAI function-calling / Agents SDK tool specs.

Two ways to use this file:
  (A) FUNCTION_SPECS — raw JSON tool schemas for the Chat Completions / Responses
      `tools=[...]` param (works with any OpenAI-compatible client).
  (B) OpenAI Agents SDK @function_tool wrappers (pip install openai-agents).

Both point at the LIVE Bay Run service. Descriptions are written for model
tool-selection. export BAY_RUN_TOKEN=...  (get one at https://huggingbay.xyz)
"""
import os
import httpx

BASE = os.environ.get("BAY_RUN_BASE_URL", "https://bay-run-mvp-zfmlsu2yla-uc.a.run.app")
HEADERS = {"Authorization": f"Bearer {os.environ.get('BAY_RUN_TOKEN', '')}",
           "Content-Type": "application/json"}


def _post(path: str, payload: dict, timeout: int = 120) -> dict:
    with httpx.Client(timeout=timeout) as c:
        r = c.post(f"{BASE}{path}", headers=HEADERS, json=payload)
        r.raise_for_status()
        return r.json()


# ---------------------------------------------------------------------------
# (A) Raw tool schemas for OpenAI-compatible function calling.
# ---------------------------------------------------------------------------
FUNCTION_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "bay_run_find_specialist",
            "description": (
                "Find the best small open specialist model for a narrow, high-volume task "
                "(embeddings, reranking, classification, extraction, routing) and prove it on "
                "the caller's own examples — cheaper per-call than a frontier API. Chains "
                "catalog-search + on-your-data bake-off and returns the winning model id, "
                "scorecard, and serving endpoint. Prefer this over guessing a model name."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "Plain-language task description."},
                    "my_examples": {
                        "type": "array",
                        "description": "Labeled examples, each {query, positive, negatives:[...]}.",
                        "items": {"type": "object"},
                    },
                    "kind": {"type": "string", "enum": ["embedding", "rerank", "any"], "default": "any"},
                },
                "required": ["task", "my_examples"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bay_run_discover_models",
            "description": (
                "Search a 147K-model catalog for candidate open specialist models for a task. "
                "Returns ranked candidate model ids (mirrored-first = instant serve). Candidates "
                "only — feed them to bay_run_eval_models."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "kind": {"type": "string",
                             "enum": ["embedding", "llm", "vision", "audio", "tool", "agent", "any"],
                             "default": "any"},
                    "limit": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bay_run_eval_models",
            "description": (
                "Bake off candidate models on the caller's labeled data; return a ranked "
                "scorecard + winner. A public leaderboard rank does NOT predict your-domain fit."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "enum": ["embedding", "rerank"]},
                    "models": {"type": "array", "items": {"type": "string"}},
                    "dataset": {"type": "array", "items": {"type": "object"},
                                "description": "Items each {query, positive, negatives:[...]}."},
                },
                "required": ["task", "models", "dataset"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bay_run_embed",
            "description": ("Embed text with any open embedding model, served instantly "
                            "(OpenAI-compatible). Drop-in for RAG/semantic-search pipelines."),
            "parameters": {
                "type": "object",
                "properties": {
                    "model": {"type": "string", "description": "Any HF embedding model id."},
                    "input": {"description": "String or list of strings."},
                },
                "required": ["model", "input"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bay_run_rerank",
            "description": ("Rerank documents by relevance to a query using any open cross-encoder "
                            "— sharpen RAG/search precision after a noisy top-k retrieval."),
            "parameters": {
                "type": "object",
                "properties": {
                    "model": {"type": "string", "description": "Any HF reranker/cross-encoder id."},
                    "query": {"type": "string"},
                    "documents": {"type": "array", "items": {"type": "string"}},
                    "top_n": {"type": "integer"},
                },
                "required": ["model", "query", "documents"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bay_run_extract",
            "description": (
                "Turn messy HTML/text (e.g. a scraper/Firecrawl dump) into schema-guided "
                "STRUCTURED JSON using a small CPU-served generative specialist — the reliable "
                "extraction layer after scrapers, cheap at volume. Returns {data, json_valid, "
                "raw}; this is BEST-EFFORT — always check json_valid before trusting data. "
                "Deterministic (greedy)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "model": {"type": "string",
                              "description": "A generative HF id (or the winner from bay_run_find_specialist)."},
                    "content": {"type": "string", "description": "Raw HTML/text to extract from."},
                    "schema": {"type": "object",
                               "description": "OpenAI-style JSON schema {properties, required} pinning fields."},
                    "instructions": {"type": "string", "description": "Optional extra guidance."},
                },
                "required": ["model", "content"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Dispatch: map a tool call name+args to the live endpoint.
# ---------------------------------------------------------------------------
def call_bay_run_tool(name: str, args: dict) -> dict:
    if name == "bay_run_find_specialist":
        kind = args.get("kind", "any")
        disc = _post("/v1/discover", {"query": args["task"], "kind": kind, "limit": 5}, timeout=60)
        # discover.py contract: candidates[].model  (NOT results[].id)
        cands = [c.get("model") for c in disc.get("candidates", [])]
        cands = [c for c in cands if c][:5]
        et = "rerank" if kind == "rerank" else "embedding"
        ev = _post("/v1/eval", {"task": et, "models": cands,
                                "dataset": args["my_examples"], "k": 5}, timeout=600)
        return {"winner": ev.get("winner"), "scorecard": ev.get("scorecard"),
                "serve_endpoint": f"{BASE}/v1/{'rerank' if et=='rerank' else 'embeddings'}"}
    if name == "bay_run_discover_models":
        return _post("/v1/discover", {"query": args["query"], "kind": args.get("kind", "any"),
                                      "limit": args.get("limit", 5)}, timeout=60)
    if name == "bay_run_eval_models":
        return _post("/v1/eval", {"task": args["task"], "models": args["models"],
                                  "dataset": args["dataset"], "k": 5}, timeout=600)
    if name == "bay_run_embed":
        return _post("/v1/embeddings", {"model": args["model"], "input": args["input"]})
    if name == "bay_run_rerank":
        p = {"model": args["model"], "query": args["query"], "documents": args["documents"]}
        if args.get("top_n") is not None:
            p["top_n"] = args["top_n"]
        return _post("/v1/rerank", p)
    if name == "bay_run_extract":
        if args.get("schema") is not None:
            rf = {"type": "json_schema", "json_schema": {"schema": args["schema"]}}
        else:
            rf = {"type": "json_object"}
        msg = (args["instructions"] + "\n\n" if args.get("instructions") else "") + args["content"]
        resp = _post("/v1/chat/completions", {
            "model": args["model"], "messages": [{"role": "user", "content": msg}],
            "response_format": rf, "temperature": 0.0,
        }, timeout=600)
        return {"model": args["model"], "data": resp.get("parsed"),
                "json_valid": resp.get("json_valid"),
                "raw": resp["choices"][0]["message"]["content"], "usage": resp.get("usage")}
    raise ValueError(f"unknown tool {name}")


# ---------------------------------------------------------------------------
# (B) OpenAI Agents SDK variant (optional). Uncomment if using `openai-agents`.
# ---------------------------------------------------------------------------
# from agents import function_tool
#
# @function_tool
# def find_specialist_for_task(task: str, my_examples: list, kind: str = "any") -> dict:
#     """Find + prove the best small open specialist model for a task on your examples."""
#     return call_bay_run_tool("bay_run_find_specialist",
#                              {"task": task, "my_examples": my_examples, "kind": kind})
