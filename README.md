# Bay Run — find, prove, and serve the right task-specialist model

Agents don't need one giant model for everything — they need the *right small specialist*
for each narrow job (embeddings, reranking, classification, extraction, transcription),
proven on *their* data, served instantly. Bay Run is that loop — OpenAI-compatible and MCP-native:

> **discover → eval → serve**  (over a catalog of 147K models, mirrored-first)

- **Live:** `https://bay-run-mvp-zfmlsu2yla-uc.a.run.app`
- **Remote MCP:** `https://bay-run-mvp-zfmlsu2yla-uc.a.run.app/mcp/`  (streamable-HTTP; add to any MCP client)
- **Public demo token (rate-limited, try instantly):** `bayrun-demo-AS4XgfRmTHgNXRlpuP19zKeMxbcShyvP`

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
python demo.py          # ships with the public demo token; runs discover → eval → serve → cost live
```

- [`demo/README.md`](./demo/README.md) — the captured scorecard, latencies, and cost table.
- [`demo/mcp-quickstart.md`](./demo/mcp-quickstart.md) — give your agent the tools in one command.
- [`demo/firecrawl-to-bay-run.md`](./demo/firecrawl-to-bay-run.md) — scrape with Firecrawl, run the specialist here.
- [`demo/wrappers/`](./demo/wrappers/) — drop-in LangChain / LlamaIndex / OpenAI-Agents adapters.

## 30-second try
```bash
curl -s https://bay-run-mvp-zfmlsu2yla-uc.a.run.app/v1/discover -H "authorization: Bearer bayrun-demo-AS4XgfRmTHgNXRlpuP19zKeMxbcShyvP" -H "content-type: application/json" \
  -d '{"query":"multilingual sentence embeddings","kind":"embedding","limit":5}'
```

## Point any OpenAI client at it
```python
from openai import OpenAI
client = OpenAI(base_url="https://bay-run-mvp-zfmlsu2yla-uc.a.run.app/v1", api_key="bayrun-demo-AS4XgfRmTHgNXRlpuP19zKeMxbcShyvP")
client.embeddings.create(model="BAAI/bge-small-en-v1.5", input=["hello"])
```

## MCP (agent-callable)
Add the remote server `https://bay-run-mvp-zfmlsu2yla-uc.a.run.app/mcp/` to your MCP client (Bearer auth). 9 tools:
`find_specialist_for_task` (discover→eval→serve on your labeled data), `request_specialist` (serve-or-capture — returns a serve pointer if a specialist exists, else records your demand), `route` (runtime auto-router — no examples, picks a specialist per-request), `discover_models`, `eval_models`, `embed`, `rerank`, `classify` (guardrail/moderation/sentiment/intent, or zero-shot via candidate_labels + an NLI model), `extract` (HTML/text → schema-guided JSON).

Served from a content-addressed, quarantine-gated mirror. Neutral — it helps you pick the
model that wins on *your* data, not sell you one. Backend is closed; this repo is the
public manifest + connector.
