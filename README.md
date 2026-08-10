# Bay Run — find, prove, and serve the right task-specialist model

Agents don't need one giant model for everything — they need the *right small specialist*
for each narrow job (embeddings, reranking, classification, extraction, transcription),
proven on *their* data, served instantly. Bay Run is that loop — OpenAI-compatible and MCP-native:

> **discover → eval → serve**  (over a catalog of 147K models, mirrored-first)

- **Live:** `https://bay-run-mvp-889989800693.us-central1.run.app`
- **Remote MCP:** `https://bay-run-mvp-889989800693.us-central1.run.app/mcp/`  (streamable-HTTP; add to any MCP client)
- **Autonomous OAuth:** `POST /oauth/token` with `{"grant_type":"client_credentials"}`; no client secret or browser
- **Agent discovery:** `/.well-known/mcp/server-card.json`, `/llms.txt`, and `/.well-known/oauth-protected-resource`

## ⭐ Flagship showcase — the Semantic Intent-Router
[**`flagship/`**](./flagship/) is the headline proof of the thesis. Every agent framework routes each
user message to the right handler — usually with a `$$` frontier-LLM call. The flagship does the same
routing with a **33M-param open embedder Bay Run picks + serves** (`thenlper/gte-small`, proven on
labeled intents by a live bake-off): **95.8% routing accuracy on unseen messages, ~140 ms warm,
~14× cheaper than a GPT-4o-mini intent call** — every number captured from the live service.

```bash
git clone https://github.com/barneywohl/bay-run && cd bay-run/flagship
pip install -r requirements.txt && python router_demo.py
```
See [`flagship/README.md`](./flagship/README.md) for the scorecard + cost table, and
[`flagship/more-specialists.md`](./flagship/more-specialists.md) for four more agent sub-tasks (RAG
rerank, multilingual routing, dedup, semantic cache), each with a verified-servable tiny specialist.

## Runnable demo — the whole loop in one command
[**`demo/`**](./demo/) is a self-contained killer demo: it routes support tickets with a 22M-param open
embedder picked by a bake-off on labeled data — **~26× cheaper than a GPT-4o-mini classification baseline**,
at equal-or-lower latency, open weights, no lock-in. Every number is captured from the live service.

```bash
git clone https://github.com/barneywohl/bay-run && cd bay-run/demo
pip install -r requirements.txt
python demo.py          # self-mints short-lived OAuth; runs discover → eval → serve → cost live
```

- [`demo/README.md`](./demo/README.md) — the captured scorecard, latencies, and cost table.
- [`demo/mcp-quickstart.md`](./demo/mcp-quickstart.md) — give your agent the tools in one command.
- [`demo/firecrawl-to-bay-run.md`](./demo/firecrawl-to-bay-run.md) — scrape with Firecrawl, run the specialist here.
- [`demo/wrappers/`](./demo/wrappers/) — drop-in LangChain / LlamaIndex / OpenAI-Agents adapters.

## 30-second try
```bash
BASE=https://bay-run-mvp-889989800693.us-central1.run.app
TOKEN=$(curl -fsS "$BASE/oauth/token" -H 'content-type: application/json' \
  -d '{"grant_type":"client_credentials"}' | python -c 'import json,sys; print(json.load(sys.stdin)["access_token"])')
curl -s "$BASE/v1/discover" -H "authorization: Bearer $TOKEN" -H "content-type: application/json" \
  -d '{"query":"multilingual sentence embeddings","kind":"embedding","limit":5}'
```

## Point any OpenAI client at it
```python
import requests
from openai import OpenAI

base = "https://bay-run-mvp-889989800693.us-central1.run.app"
token = requests.post(f"{base}/oauth/token", json={"grant_type": "client_credentials"}).json()["access_token"]
client = OpenAI(base_url=f"{base}/v1", api_key=token)
client.embeddings.create(model="BAAI/bge-small-en-v1.5", input=["hello"])
```

## MCP (agent-callable)
Add the remote server `https://bay-run-mvp-889989800693.us-central1.run.app/mcp/` to your MCP client. It exposes 20 tools:
`try_bay_run`, `find_specialist_for_task`, `request_specialist`, `route`, `discover_models`, `eval_models`, `embed`, `rerank`, `classify`, `extract`, `summarize`, `rag_search`, `memory_context`, `speed_test`, `remember`, `recall`, `forget`, `calculate`, `validate_json`, and `resolve_link`.

MCP discovery and `try_bay_run` are public. Other calls use OAuth bearer auth. Persist the OAuth refresh token so the same private memory principal follows the agent across renewals and harnesses.

Served from a content-addressed, quarantine-gated mirror. Neutral — it helps you pick the
model that wins on *your* data, not sell you one. Backend is closed; this repo is the
public manifest + connector.
