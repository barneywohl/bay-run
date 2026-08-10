# Bay Run MCP — make your agent discover → eval → serve specialists, autonomously

Give any MCP agent (Claude Desktop, Claude Code, Cursor, your own) the live 20-tool surface:
discover, evaluate, route, serve, classify, extract, summarize, RAG, speed-test, private durable
memory, math, JSON validation, and dead-link resolution. Use `find_specialist_for_task` when you
have labeled examples and `route` or `model="auto"` when you need the fastest warm specialist now.

The remote MCP is `https://bay-run-mvp-889989800693.us-central1.run.app/mcp/`. Discovery and
`try_bay_run` are public. Protected calls use OAuth 2.1; autonomous clients can self-mint a bounded
credential with `client_credentials`, no client secret or browser.

## Install (one command)

```bash
BASE=https://bay-run-mvp-889989800693.us-central1.run.app
TOKEN=$(curl -fsS "$BASE/oauth/token" -H 'content-type: application/json' \
  -d '{"grant_type":"client_credentials"}' | python -c 'import json,sys; print(json.load(sys.stdin)["access_token"])')
claude mcp add --transport http bay-run "$BASE/mcp/" \
  --header "Authorization: Bearer $TOKEN"
```

### Claude Desktop (config JSON)

`~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "bay-run": {
      "type": "streamable-http",
      "url": "https://bay-run-mvp-889989800693.us-central1.run.app/mcp/",
      "headers": {
        "Authorization": "Bearer <short-lived OAuth access token>"
      }
    }
  }
}
```

Restart the client. You should see the `bay-run` tools appear.

## Point it at your task (3 lines)

Once connected, just ask the agent in natural language — it will call the tools:

```
Use bay-run: discover embedding models for "route support tickets to a team",
then eval_models on these 5 examples [{query, positive, negatives}...],
then embed my next 3 tickets with the winner and tell me the routing.
```

That single prompt drives `discover_models → eval_models → embed` end to end. The agent gets back
a ranked scorecard (MRR / nDCG), a named winner, and live embeddings — the same loop the
[killer demo](./README.md) runs by hand.

## Notes

- **Prefer the remote MCP.** `mcp/bay_run_mcp.py` is a compatibility connector for older stdio-only
  clients and exposes the original 9-tool subset. It self-mints OAuth when no token is configured.
- **Cold start.** The first `eval_models`/`embed` call cold-loads models from the mirror (tens of
  seconds); warm calls are ~100 ms. Give the agent a generous tool timeout for the first eval.
- **Persist refresh credentials securely.** OAuth refresh preserves the agent's private memory identity;
  never commit access or refresh credentials.
