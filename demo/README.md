# Killer demo: a 22M-param specialist routes your support tickets — ~26× cheaper than a frontier LLM

**One narrow, high-volume job (support-ticket routing). One tiny open model, picked by a bake-off on
*your* labeled data, served instantly from an OpenAI-compatible endpoint. No GPU, no packaging, no lock-in.**

This folder is the whole thing. Clone it, set one env var, run one file. Every number below was
captured from the **live** Bay Run service — the only thing that isn't a real API call is the frontier
cost baseline, which is *computed* from published pricing (we don't spend your money to make a point).

```
discover  →  eval (bake-off on YOUR data)  →  run (serve the winner)  →  compare ($ / latency)
```

---

## Run it (<5 min)

```bash
pip install -r requirements.txt
python demo.py                          # runs against the live service with the built-in public demo token
```

- **Zero-config:** `demo.py` ships with a **public, rate-limited demo token** so it runs in one command.
  It's low-limit on purpose — for real volume, mint your own at `https://huggingbay.xyz/tokens` and
  `export BAY_RUN_TOKEN=<your token>` to override. Never commit a private token.
- Base URL is baked in: `https://bay-run-mvp-zfmlsu2yla-uc.a.run.app` (override with `BAY_RUN_BASE_URL`).
- `dataset.json` is a **support-ticket routing** set of `{query, positive, negatives}`. Swap it for your
  own tickets + your own taxonomy — that's the point: the winner is chosen on *your* data.
- First run is slow (the service is scale-to-zero and cold-loads each candidate from the mirror). Warm
  calls are ~100 ms. See the cold-start note at the bottom.

---

## What actually happened (captured live, 2026-08-08)

### 1. discover — candidates, not winners

`POST /v1/discover` returned 6 mirrored embedding candidates in **393 ms**. The top hits are the popular
`thenlper/gte-*` family — but a leaderboard/download count is **not** proof they win on *your* tickets.
That's the whole reason the next step exists. (Full JSON: `captured/discover.json`.)

### 2. eval — bake-off on YOUR labeled data (the money shot)

`POST /v1/eval` ran 4 candidates over 8 labeled ticket→label examples and returned a ranked scorecard.
Primary metric is **MRR** (mean reciprocal rank of the correct label). Wall time for the whole bake-off,
warm: **~7.5 s**.

| model | MRR | nDCG@3 | verified | note |
|---|---|---|---|---|
| **sentence-transformers/all-MiniLM-L6-v2** (22M params, ~90 MB) | **1.000** | **1.000** | ✅ | **winner** |
| BAAI/bge-small-en-v1.5 | 1.000 | 1.000 | ✅ | ties on quality, ~2× slower to load |
| thenlper/gte-small | 1.000 | 1.000 | ✅ | ties on quality, larger |
| sentence-transformers/average_word_embeddings_glove.6B.300d | 0.854 | 0.891 | ❌ | weak baseline — eval **fails it** |

Two things this proves:
- **The eval discriminates.** A deliberately weak GloVe baseline drops to **MRR 0.854** and gets two
  queries wrong (`per_query_rank` had a 2 and a 3). An eval that can't fail anything is worthless; this one can.
- **You don't need a big model for this job.** A 22M-param model ties the field at perfect MRR, so the
  tiebreak that matters is **cost + latency** — where the smallest model wins. The winner is the *cheapest
  thing that's provably good enough on your data*, not the top of a leaderboard.

Raw scorecard JSON: `captured/eval_scorecard.json`. Full console transcript: `captured/demo_run.txt`.

> Honest note: `intfloat/multilingual-e5-small` was in an earlier candidate list and **errored** in the
> harness (a tokenizer/processor-class issue in that model's mirror cache). The eval reported the error
> per-model and kept ranking the rest instead of failing the whole run — which is the behavior you want.

### 3. run — serve the winner, route unseen tickets

The winner is served straight from `POST /v1/embeddings` (OpenAI-shaped). We embed 8 labeled example
tickets once as routing anchors, then classify brand-new tickets by nearest anchor:

| unseen ticket | routed to | latency |
|---|---|---|
| "hey my card got declined but you still show my plan as active?" | Billing: charged after cancellation | 96 ms |
| "app just white-screens after the latest update, nothing loads" | Bug (login/500 family) | 94 ms |
| "interested in rolling this out to 300 people, do you support SAML?" | Sales: enterprise security review and SSO | 98 ms |
| "please wipe my account and everything you have on me" | Privacy: GDPR data deletion request | 98 ms |

**Median warm round-trip latency (incl. public internet): 98 ms.** Three of four land on the exact ideal
label; the white-screen ticket lands the right *team* (Bug) even though there's no dedicated "app won't
load" label in this tiny taxonomy. In production you sharpen the ambiguous cases with a second-stage
**reranker** (`POST /v1/rerank`, also live — `cross-encoder/ms-marco-MiniLM-L-6-v2` warm ≈ 96 ms) over the
top-k candidate labels.

### 4. compare — cost at 1,000,000 routing requests / month

No paid API was called. Frontier numbers are computed from published pricing (sources below).

| approach | $ / 1M requests | assumptions |
|---|---:|---|
| **GPT-4o-mini** — LLM classification | **$69.00** | 400 input tok (system + ~15 category defs + ticket) + 15 output tok/req; $0.15/1M in, $0.60/1M out |
| OpenAI `text-embedding-3-small` — embed only | $2.00 | 100 tok/ticket × $0.02/1M; *you still host the vector search* |
| **Bay Run winner** (`all-MiniLM-L6-v2`, CPU) | **~$2.70** | measured 0.093 s warm/req on 1 vCPU + 2 GiB Cloud Run compute |

**Bay Run is ~26× cheaper than the GPT-4o-mini classification approach** — the way most indie/small-team
devs route tickets today — at equal-or-lower latency, **with open weights (no lock-in) and a model proven
on your data**. Even against the cheapest frontier *embedding* API (which is genuinely cheap on tokens),
Bay Run gives you: open weights you can pull and run anywhere, a reranker in the same place, and an
eval that told you *which* model to trust — instead of a metered endpoint you can't move off.

Pricing sources (Aug 2026):
- GPT-4o-mini $0.15/1M in, $0.60/1M out — https://pricepertoken.com/pricing-page/model/openai-gpt-4o-mini
- text-embedding-3-small $0.02/1M — https://pricepertoken.com/embedding/model/openai-text-embedding-3-small
- Cloud Run compute rates (approx, us) — https://cloud.google.com/run/pricing

---

## The honest asterisks

- **Cold start is real.** Scale-to-zero + first-time model load from the mirror can take 20–50 s on the
  very first call (we saw a 32 s first embedding call, and one candidate cold-loaded in 52 s inside an
  eval). Warm steady-state is ~100 ms. For latency-critical paths you keep one instance warm.
- **This is retrieval/classification, not generative extraction.** The live endpoints do embeddings +
  rerank. HTML→structured-JSON *generative* extraction is on the roadmap, not in this demo — see the roadmap note in the repo README.
- **The cost win is vs the naive LLM-classification pattern**, which is exactly what the target persona is
  overpaying for. Against a tuned embedding-API + self-hosted search you're competing on lock-in,
  neutrality, and the eval — not raw token price.
- **`verified` in the scorecard is a provenance stub** (curated-list membership) in this MVP; the real
  Ed25519 manifest verdict is wired separately on huggingbay.xyz.

## Files

| file | what |
|---|---|
| `demo.py` | the whole loop, ~180 lines, hits the live service |
| `dataset.json` | the labeled bake-off set — **replace with yours** |
| `requirements.txt` | just `requests` |
| `captured/demo_run.txt` | full console transcript of the live run |
| `captured/eval_scorecard.json` | raw eval response |
| `captured/discover.json` | raw discover response |
