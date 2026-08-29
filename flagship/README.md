# Archived benchmark — Semantic Intent Router

This directory preserves an earlier specialist-routing benchmark. It is not the current Bay Run
activation path or default product surface. For a new integration, use the live
[Bay Run quickstart](https://run.huggingbay.xyz/quickstart) and the three-tool MCP front door:
`coprocessor`, `run_pin`, and `solve_task`.

**Your agent routes every incoming message with a `$$` frontier-LLM call. Here's the exact
same routing done by a tiny open embedder that Bay Run picks and serves — ~14× cheaper,
~140 ms warm, 95.8% accurate on unseen messages, open weights, no lock-in.**

This archived benchmark evaluates one narrow sub-task: taking a user message and routing it to
the right handler, tool, or skill. It compares a specialist approach with an illustrative
frontier-model baseline; it does not describe the current default onboarding flow.

> **discover → eval → serve** — a 33M-param open embedder (`thenlper/gte-small`), *proven on
> our labeled intents by a live bake-off*, then served from Bay Run's `/v1/embeddings`.

Everything below is captured from the **live** service. Reproduce it in one command.

## Run it

```bash
git clone https://github.com/barneywohl/bay-run && cd bay-run/flagship
pip install -r requirements.txt
python router_demo.py          # self-mints bounded OAuth; runs the whole loop live
```

The script (`router_demo.py`) is paced for the bounded OAuth demo tier. Set `BAY_RUN_TOKEN`
only when your harness already manages a credential.

## What the run proves (captured, live)

| | |
|---|---|
| Task | Route unseen user messages to 1 of **10 intents** (billing, tech_support, cancel, sales, refund, docs, escalate, account, security, feedback) |
| Eval winner (on our data) | **`thenlper/gte-small`** — MRR **0.912**, nDCG@5 **0.934** |
| Routing accuracy | **23/24 = 95.8%** on held-out, unseen messages |
| Warm serving latency | **median ~143 ms**, p95 ~152 ms (round-trip incl. network from a laptop) |
| Cost vs frontier LLM | **~14× cheaper** than a GPT-4o-mini intent call (see table) |

### The eval scorecard (why gte-small, not the popular pick)

```
model                                                       MRR   nDCG@5  verified
----------------------------------------------------------------------------------
thenlper/gte-small                                        0.912    0.934      True   <- winner
sentence-transformers/all-MiniLM-L6-v2                    0.877    0.908      True
BAAI/bge-small-en-v1.5                                    0.875    0.907      True
sentence-transformers/average_word_embeddings_glove.6B.300d  0.795  0.847     False  <- weak baseline, fails verify
```

`BAAI/bge-small-en-v1.5` has **61M downloads** vs gte-small's ~322K — popularity would have
picked the *worse* model for our intents. That's the whole point of the eval step: MTEB rank
and download counts don't predict *your-domain* performance. The GloVe averager is included on
purpose so you can see the eval **fail** something — a trustworthy bake-off must be able to.

### Cost at 1,000,000 routing requests / month

```
approach                                              $ / 1M req
----------------------------------------------------------------
GPT-4o-mini  (LLM intent classification)                  57.30
Bay Run winner: gte-small (self-host CPU)                  4.16   <- ~14x cheaper
OpenAI text-embedding-3-small  (embeddings path)           0.40   <- the path itself is ~140x cheaper
```

**Assumptions (stated, honest):** LLM route = ~350 input tokens (instructions + 10 intent defs +
message) + 8 output tokens at GPT-4o-mini published pricing ($0.15/1M in, $0.60/1M out). Bay Run CPU
priced at the **full measured round-trip latency** × Cloud Run compute ($0.000024/vCPU-s +
$0.0000025/GiB-s, 1 vCPU + 2 GiB) — deliberately *conservative*, since round-trip includes network
idle, not just server compute, so real compute cost is lower. No paid API was called; the frontier
numbers are computed, the Bay Run scorecard / accuracy / latency are measured.

**The one honest miss:** *"someone changed my password and i'm now locked out completely"* routed to
`account` when the gold label was `security` — a genuinely ambiguous message that sits between the two
intents. 23/24 with zero per-intent tuning is the real number.

## 30-second try (copy-paste)

Discover a routing embedder:
```bash
BASE=https://bay-run-mvp-889989800693.us-central1.run.app
TOKEN=$(curl -fsS "$BASE/oauth/token" -H 'content-type: application/json' \
  -d '{"grant_type":"client_credentials"}' | python -c 'import json,sys; print(json.load(sys.stdin)["access_token"])')
curl -s "$BASE/v1/discover" \
  -H "authorization: Bearer $TOKEN" \
  -H "content-type: application/json" \
  -d '{"query":"embed short user messages to route them to the right intent","kind":"embedding","limit":5}'
```

Serve the winner (OpenAI-compatible):
```bash
curl -s "$BASE/v1/embeddings" \
  -H "authorization: Bearer $TOKEN" \
  -H "content-type: application/json" \
  -d '{"model":"thenlper/gte-small","input":["please stop billing me, i'\''m done"]}'
```

Or point any OpenAI client at `.../v1`, or add the MCP server `.../mcp/` to your agent.
The remote MCP publishes the current default tools `coprocessor`, `run_pin`, and `solve_task`;
advanced capabilities are progressively disclosed through the live references.

## Files

- [`router_demo.py`](./router_demo.py) — one-command, live discover → eval → serve → route → cost.
- [`intents.json`](./intents.json) — the 10-intent taxonomy + held-out unseen test messages. Swap for yours.
- [`captured/`](./captured/) — the real transcript ([`demo_run.txt`](./captured/demo_run.txt)),
  eval scorecard, discover result, and machine-readable [`summary.json`](./captured/summary.json).
- [`more-specialists.md`](./more-specialists.md) — four more narrow agent tasks + the tiny specialist
  Bay Run serves for each (every model verified servable on the live endpoint).

---

**The thesis, made concrete:** don't call a frontier LLM for a job a tiny specialist does better and
cheaper. Bay Run finds the specialist, proves it on *your* data, and serves it — over 147K open
models, mirrored-first, no lock-in.
