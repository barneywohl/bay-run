# The breadth: a right tiny model per task

Intent routing is one sub-task. The thesis is general — **most narrow jobs an agent farms out to a
frontier LLM have a small open specialist that does that one job better and far cheaper.** Below are
four more, each with a specialist **verified servable on the live Bay Run endpoint** (checked
2026-08-08 via `/v1/rerank` and `/v1/embeddings` returning `200` with real scores/vectors).

| Agent sub-task | Frontier-LLM way (what it costs you) | Tiny specialist Bay Run serves | Why the small model wins |
|---|---|---|---|
| **RAG answer reranking** — reorder the top-K retrieved chunks before you stuff the prompt | An LLM "grade these passages" pass per query — extra tokens + latency on every RAG turn | **`BAAI/bge-reranker-base`** (`/v1/rerank`) ✅ 200 — scored the correct password-reset doc **0.997** vs 0.009/0.001 for distractors | A cross-encoder is *built* for query-document relevance; one call ranks all candidates, no generation tokens, deterministic scores |
| **Multilingual message routing / retrieval** — same routing, but users write in any language | A frontier LLM to translate-then-classify, or a big multilingual model | **`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`** (`/v1/embeddings`) ✅ 200, 384-dim — embeds `"¿cómo restablezco mi contraseña?"` directly | 50+ languages share one vector space; route by nearest centroid with no translation hop, at embedding cost |
| **Near-duplicate / dedup detection** — collapse repeated tickets, dedupe a crawl, cluster feedback | An LLM "are these the same?" call per pair — quadratic and pricey | **`sentence-transformers/all-MiniLM-L6-v2`** (`/v1/embeddings`) ✅ 200, 384-dim, ~22M params | Embed once, compare with cosine; O(n) embeddings + cheap vector ops instead of O(n²) LLM calls |
| **Semantic cache / repeat-question hit** — answer FAQs and repeat asks without hitting the LLM | Every repeat question pays full LLM price again | **`BAAI/bge-small-en-v1.5`** (`/v1/embeddings`) ✅ 200, 384-dim (proven in the flagship eval, MRR 0.875) | Embed the incoming question, match against cached Q→A vectors; a hit skips the LLM call entirely |

A **second reranker**, `cross-encoder/ms-marco-MiniLM-L-6-v2`, is also verified servable
(`/v1/rerank` → 200) — so you can *eval two rerankers head-to-head* on your own chunks before
committing, exactly like the flagship does for embedders.

Need a bigger vector when recall matters (clustering, large-corpus retrieval)?
**`mixedbread-ai/mxbai-embed-large-v1`** is servable too (`/v1/embeddings` → 200, **1024-dim**) — same
endpoint, no repackaging. The point isn't "always the smallest model"; it's "the *right-sized* model
for the job, proven on your data."

## Honesty note — this is exactly why the `eval` step exists

Not every catalog entry loads. Two models listed in `/v1/models` currently **fail to serve** on the
live endpoint:

- `intfloat/multilingual-e5-small` → `400` "Unrecognized processing class … can't instantiate a tokenizer"
- `jinaai/jina-reranker-v1-tiny-en` → `400` `attn_implementation="torch"` not supported

That's the case *for* the discover → **eval** → serve loop: you find out a candidate is broken (or just
worse on your data) in a bake-off **before** you wire it into production — not at 2am in prod. Every
model recommended in the table above was verified to actually return real results on the live service.
