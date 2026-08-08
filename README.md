# Bay Run — find, prove, and serve the right task-specialist model

Agents don't need one giant model for everything — they need the *right small specialist*
for each narrow job (embeddings, reranking, classification, extraction, transcription),
proven on *their* data, served instantly. Bay Run is that loop — OpenAI-compatible and MCP-native:

> **discover → eval → serve**  (over a catalog of 147K models, mirrored-first)

- **Live:** `https://bay-run-mvp-zfmlsu2yla-uc.a.run.app`
- **Remote MCP:** `https://bay-run-mvp-zfmlsu2yla-uc.a.run.app/mcp/`  (streamable-HTTP; add to any MCP client)
- **Public demo token (rate-limited, try instantly):** `bayrun-demo-AS4XgfRmTHgNXRlpuP19zKeMxbcShyvP`

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
Add the remote server `https://bay-run-mvp-zfmlsu2yla-uc.a.run.app/mcp/` to your MCP client (Bearer auth). Tools:
`discover_models`, `eval_models`, `embed`, `rerank`, `extract`, `find_specialist_for_task`.

Served from a content-addressed, quarantine-gated mirror. Neutral — it helps you pick the
model that wins on *your* data, not sell you one. Backend is closed; this repo is the
public manifest + connector.
