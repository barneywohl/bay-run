# Bay Run MCP quickstart

This companion quickstart follows Bay Run’s current public MCP front door. Use the
[canonical live quickstart](https://run.huggingbay.xyz/quickstart) for auth and request details that
may change.

The canonical Streamable HTTP MCP URL is **<https://run.huggingbay.xyz/mcp/>**. Its default tools are
exactly, in order: `coprocessor`, `run_pin`, `solve_task`.

## Connect

Mint a resource-bound demo bearer, then add the remote server to an MCP client:

```bash
BASE=https://run.huggingbay.xyz
TOKEN=$(curl -fsS -X POST "$BASE/oauth/token" \
  -H 'Content-Type: application/json' \
  -d '{"grant_type":"urn:bay-run:grant-type:demo","scope":"mcp:demo","resource":"https://run.huggingbay.xyz/mcp/"}' \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["access_token"])')

claude mcp add --transport http bay-run "$BASE/mcp/" \
  --header "Authorization: Bearer $TOKEN"
```

For a desktop client, the equivalent configuration is:

```json
{
  "mcpServers": {
    "bay-run": {
      "type": "streamable-http",
      "url": "https://run.huggingbay.xyz/mcp/",
      "headers": {
        "Authorization": "Bearer <resource-bound demo bearer>"
      }
    }
  }
}
```

## Start with `coprocessor`

The first call should be the bounded Guard-first coprocessor. It never generates text or executes
tools; add `documents` only when retrieved chunks need bounded post-SAFE reranking.

```bash
curl -fsS -X POST "$BASE/mcp/" \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"coprocessor","arguments":{"user_text":"<untrusted request>"}}}'
```

Use `run_pin` when a known canonical Pin matches the task. Use `solve_task` only as the open-ended
fallback, with `task_description` and `input`. Read the action and evidence before allowing a
caller-owned generation or tool call to continue.

The checked-in stdio wrapper is compatibility-only and does not define the current remote MCP
tool contract. Advanced REST and MCP capabilities are progressively disclosed through the [OpenAPI
contract](https://run.huggingbay.xyz/openapi.json), [advanced MCP tools](https://run.huggingbay.xyz/.well-known/mcp/advanced-tools.json),
and [full LLM reference](https://run.huggingbay.xyz/llms-full.txt).
