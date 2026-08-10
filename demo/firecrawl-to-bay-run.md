# Firecrawl → Bay Run: scrape the web, then let a specialist do the inference

**Firecrawl gets you the *content*. Bay Run runs the *specialist model* over it — cheaply, on CPU,
proven on your data.** They're not competitors; they're two halves of the same pipeline. Firecrawl (and
Exa, E2B, and friends) sit at the *acquisition/execution* layer. Bay Run is the **specialist-inference
layer downstream**: embed, rerank, classify, route. A "Firecrawl → Bay Run" flow is co-marketing, not
overlap.

```
Firecrawl /scrape  ──▶  clean markdown  ──▶  Bay Run  ──▶  embeddings / rerank / routing
   (get the page)         (chunks)          (the model that's cheap + proven on your data)
```

> Note: this recipe illustrates the **call shapes**. You supply a Firecrawl key for the scrape step and a
> Bay Run token for the inference step. The Bay Run calls are exactly the ones proven live in
> [`demo.py` in this folder](./README.md).

---

## Step 1 — Firecrawl scrapes a page to markdown

Firecrawl v2 scrape endpoint ([docs](https://docs.firecrawl.dev/api-reference/endpoint/scrape)):

```bash
curl -s -X POST https://api.firecrawl.dev/v2/scrape \
  -H "Authorization: Bearer $FIRECRAWL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/pricing", "formats": ["markdown"], "onlyMainContent": true}'
```

You get back clean markdown (no nav/boilerplate). Split it into chunks — sections, paragraphs, or table
rows — and hand them to Bay Run.

## Step 2 — Bay Run embeds the chunks (build a retrieval index)

```bash
BASE=https://bay-run-mvp-889989800693.us-central1.run.app
curl -s $BASE/v1/embeddings -H "authorization: Bearer <your token>" -H "content-type: application/json" \
  -d '{"model":"sentence-transformers/all-MiniLM-L6-v2",
       "input":["Team plan: $12/user/mo, up to 50 seats",
                "Enterprise: custom pricing, SSO, SAML, audit logs",
                "Free tier: 1 project, community support"]}'
```

384-dim vectors, ~100 ms warm. Store them in your vector DB. Same OpenAI-shaped response an OpenAI client
expects — zero adapter code.

## Step 3 — Bay Run reranks scraped chunks against a question (precision retrieval)

Instead of trusting raw embedding similarity, run a cross-encoder over the top chunks — sharper answers,
still CPU-cheap:

```bash
curl -s $BASE/v1/rerank -H "authorization: Bearer <your token>" -H "content-type: application/json" \
  -d '{"model":"cross-encoder/ms-marco-MiniLM-L-6-v2",
       "query":"do they support SAML single sign-on?",
       "documents":["Team plan: $12/user/mo, up to 50 seats",
                    "Enterprise: custom pricing, SSO, SAML, audit logs",
                    "Free tier: 1 project, community support"],
       "top_n":1}'
```

Returns the Enterprise/SSO chunk on top. (Verified live: warm rerank ≈ 96 ms.)

## Step 4 — or classify/route the scraped page

Crawling a list of pages (competitors, job posts, product listings, changelogs)? Embed each against your
label anchors and route it — the exact loop in the killer demo. Pick the *winning* classifier for your
taxonomy with `POST /v1/eval` first, then serve it.

---

## Why do the inference on Bay Run instead of a frontier API

For a **high-volume, narrow** step (embed every scraped chunk, classify every crawled page), a frontier
chat model is the wrong tool — you pay per token to do what a 22M-param open model does for ~1/25th the
cost (see the [cost table](./README.md#4-compare--cost-at-1000000-routing-requests--month)),
and you get **open weights with no lock-in**. Firecrawl handles the hard part (getting clean content past
anti-bot, JS, pagination). Bay Run handles the cheap-but-high-volume part (running the specialist). Use
each for what it's best at.

### Co-marketing angle
- "Scrape with Firecrawl, embed/rerank with Bay Run" is a clean tutorial neither tool competes on.
- Same pattern with **Exa** (search → embed the results), **E2B** (execute → classify the output).
- The joint pitch to a builder: *your scraper shouldn't also be your GPU bill.*

Firecrawl API reference: https://docs.firecrawl.dev/api-reference/endpoint/scrape
