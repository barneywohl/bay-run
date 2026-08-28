# Bay Run

Bay Run is the decision and execution layer for durable specialist Pins at the canonical origin: <https://run.huggingbay.xyz>.

The default MCP front door is exactly three tools, in this order: `coprocessor`, `run_pin`, `solve_task`.

## Quickstart: connect to MCP

Use the canonical Streamable HTTP MCP endpoint: **<https://run.huggingbay.xyz/mcp/>**. The live
copy-paste auth and request contract is kept at <https://run.huggingbay.xyz/quickstart>.

Mint a resource-bound demo bearer, then make the primary Guard-first call:

```bash
BASE=https://run.huggingbay.xyz
TOKEN=$(curl -fsS -X POST "$BASE/oauth/token" \
  -H 'Content-Type: application/json' \
  -d '{"grant_type":"urn:bay-run:grant-type:demo","scope":"mcp:demo","resource":"https://run.huggingbay.xyz/mcp/"}' \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["access_token"])')

curl -fsS -X POST "$BASE/mcp/" \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"coprocessor","arguments":{"user_text":"<untrusted request>"}}}'
```

`coprocessor` runs the canonical Guard first and never generates text or executes tools. Add
`documents` when retrieved chunks need bounded post-SAFE reranking. Read its action and evidence
before deciding whether a caller-owned generation or tool call may continue.

## Default MCP tools

Point an MCP client at **<https://run.huggingbay.xyz/mcp/>**. The public default surface is:

- `coprocessor` — the primary bounded Guard-first call over `user_text` and optional documents.
- `run_pin` — the direct alias for one of the four canonical Pins, using its published `{pin_id,input}` shape.
- `solve_task` — the open-ended fallback when no canonical Pin matches; provide `task_description` and `input`.

Advanced compatibility capabilities remain available through explicit live references:

- [OpenAPI](https://run.huggingbay.xyz/openapi.json)
- [advanced MCP tools](https://run.huggingbay.xyz/.well-known/mcp/advanced-tools.json)
- [full LLM reference](https://run.huggingbay.xyz/llms-full.txt)
- [SDK and framework integrations](docs/integrations.md)

## Trust and data policy

Read the canonical [data policy](https://run.huggingbay.xyz/.well-known/data-policy.json) before
sending input. The live [quickstart](https://run.huggingbay.xyz/quickstart) documents receipt
semantics, 401 recovery, and the published limits; do not infer capabilities from advanced routes.

## Hugging Bay mirror catalog

Hugging Bay’s mirror catalog is available at <https://huggingbay.xyz>. Use each artifact's published
download plan for the currently admitted files. A drop-in `HF_ENDPOINT` compatibility origin is not
part of the public contract yet, so this repository does not advertise one prematurely.

For current catalog and serving information, use the generated [Bay Run status snapshot](https://run.huggingbay.xyz/status.json).
Mirror counts and family breakdowns are intentionally not copied here because they change.
