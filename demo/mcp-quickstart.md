# Bay Run MCP — make your agent discover → eval → serve specialists, autonomously

Give any MCP agent (Claude Desktop, Claude Code, Cursor, your own) four tools:
`discover_models`, `eval_models`, `embed`, `rerank`. The agent can then find the right
open specialist for a narrow task, prove it on your labeled data, and serve it — without you
wiring any of it by hand.

The live service is authenticated, so the MCP server has to forward your bearer token. Use the
auth-enabled server in this folder: [`mcp/bay_run_mcp.py`](./mcp/bay_run_mcp.py).

## Install (one command)

```bash
pip install "mcp[cli]" httpx

claude mcp add bay-run \
  --env BAY_RUN_BASE_URL=https://bay-run-mvp-zfmlsu2yla-uc.a.run.app \
  --env BAY_RUN_TOKEN=<your token> \
  -- python /absolute/path/to/mcp/bay_run_mcp.py
```

### Claude Desktop (config JSON)

`~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "bay-run": {
      "command": "python",
      "args": ["/absolute/path/to/mcp/bay_run_mcp.py"],
      "env": {
        "BAY_RUN_BASE_URL": "https://bay-run-mvp-zfmlsu2yla-uc.a.run.app",
        "BAY_RUN_TOKEN": "<your token>"
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

- **The shipped stub has no auth.** The original `run-mvp/mcp/mcp_server.py` proxies to a *local*
  Bay Run with no token. `mcp/bay_run_mcp.py` here adds the `Authorization: Bearer` header so it
  works against the deployed service. That's the only difference.
- **Cold start.** The first `eval_models`/`embed` call cold-loads models from the mirror (tens of
  seconds); warm calls are ~100 ms. Give the agent a generous tool timeout for the first eval.
- **Never commit your token.** Keep `BAY_RUN_TOKEN` in the client's env/secrets, not in a checked-in file.
