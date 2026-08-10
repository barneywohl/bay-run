#!/usr/bin/env python3
"""
Bay Run flagship — the Semantic Intent-Router.

Every agent framework does the same sub-task on every turn: take an incoming user
message and route it to the right handler / tool / skill. The common way is a
frontier-LLM call per message (slow + $$$). This shows the SAME routing done by a
tiny open embedder that Bay Run discovers, proves, and serves — a fraction of the
cost and latency, open weights, no lock-in. That IS the whole thesis made concrete:

    route to the right small specialist instead of calling a frontier LLM.

The full loop, live against the deployed service:
    1. discover  — task -> candidate embedding specialists (147K-model catalog)
    2. eval      — bake off candidates on YOUR labeled intents -> scorecard + winner
    3. serve     — serve the winner via /v1/embeddings; route UNSEEN messages by
                   nearest-intent-centroid; report real routing accuracy + latency
    4. compare   — cost vs a frontier-LLM intent call at published pricing (computed)

Run (one command; self-mints a short-lived OAuth credential):
    pip install -r requirements.txt
    python router_demo.py
Override the token / base if you like:
    export BAY_RUN_TOKEN=<your token>
    export BAY_RUN_BASE_URL=<url>

Everything here hits the real service. The numbers you see are real.
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from pathlib import Path

import requests

BASE = os.environ.get("BAY_RUN_BASE_URL", "https://bay-run-mvp-889989800693.us-central1.run.app")

TOKEN = os.environ.get("BAY_RUN_TOKEN")
if not TOKEN:
    oauth = requests.post(
        f"{BASE}/oauth/token",
        json={"grant_type": "client_credentials"},
        timeout=30,
    )
    oauth.raise_for_status()
    TOKEN = oauth.json()["access_token"]

H = {"authorization": f"Bearer {TOKEN}", "content-type": "application/json"}
HERE = Path(__file__).parent
CAP = HERE / "captured"
CAP.mkdir(exist_ok=True)

# Candidate embedders to bake off. All are tiny, open, and mirrored (instant-serve).
# The GloVe averager is a deliberately weak baseline so you can SEE the eval separate
# winners from losers — a trustworthy eval must be able to fail something.
CANDIDATES = [
    "sentence-transformers/all-MiniLM-L6-v2",   # ~22M params, 384-dim
    "BAAI/bge-small-en-v1.5",                   # ~33M params, 384-dim
    "thenlper/gte-small",                       # ~33M params, 384-dim
    "sentence-transformers/average_word_embeddings_glove.6B.300d",  # weak baseline
]


_last_call = [0.0]
# OAuth client_credentials tokens use the bounded demo tier. Pace client-side to ~18/min.
MIN_INTERVAL = float(os.environ.get("BAY_RUN_MIN_INTERVAL", "3.3"))


def call(path: str, payload: dict, timeout: int = 240, throttle: bool = True):
    if throttle:
        wait = MIN_INTERVAL - (time.time() - _last_call[0])
        if wait > 0:
            time.sleep(wait)
    for attempt in range(6):
        t0 = time.time()
        r = requests.post(f"{BASE}{path}", headers=H, json=payload, timeout=timeout)
        dt = time.time() - t0
        _last_call[0] = time.time()
        if r.status_code == 429:  # bounded OAuth tier — back off
            back = float(r.headers.get("retry-after", 4 + attempt * 4))
            print(f"    (429 rate-limited; waiting {back:.0f}s and retrying...)")
            time.sleep(back)
            continue
        r.raise_for_status()
        return r.json(), dt
    r.raise_for_status()
    return r.json(), dt


def rule(title: str) -> None:
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb + 1e-9)


def main() -> None:
    spec = json.loads((HERE / "intents.json").read_text())
    intents = spec["intents"]
    tests = spec["test"]
    labels = [it["label"] for it in intents]
    keys = [it["key"] for it in intents]
    print(f"Bay Run @ {BASE}")
    print(f"Intents: {len(intents)}   Example utterances: {sum(len(it['train']) for it in intents)}"
          f"   Held-out test messages: {len(tests)}")

    # ------------------------------------------------------------------ 1. DISCOVER
    rule("1. DISCOVER  — task -> candidate intent-routing embedders")
    disc, dt = call("/v1/discover", {
        "query": "embed short user messages to route them to the right intent",
        "kind": "embedding", "limit": 6,
    })
    print(f"({dt*1000:.0f} ms)  {disc['count']} candidates from the 147K-model catalog:")
    for c in disc["candidates"]:
        print(f"  - {c['model']:<58} downloads={c['downloads']:>9}  mirrored={c['mirrored']}")
    print("\n  Download counts are popularity, NOT proof it routes YOUR intents. That is")
    print("  what eval is for. (Config: the winner is proven below, not guessed here.)")
    (CAP / "discover.json").write_text(json.dumps(disc, indent=2))

    # ---------------------------------------------------------------------- 2. EVAL
    rule("2. EVAL  — bake off candidates on YOUR labeled intents")
    # Build a ranking eval from the labeled utterances: for each utterance the correct
    # intent label is the positive; three other intent labels are hard negatives.
    rng = random.Random(7)
    eval_ds = []
    for it in intents:
        others = [x for x in labels if x != it["label"]]
        for utt in it["train"]:
            eval_ds.append({
                "query": utt,
                "positive": it["label"],
                "negatives": rng.sample(others, 3),
            })
    print(f"Candidates: {len(CANDIDATES)}   Ranking queries: {len(eval_ds)}")
    print("Running bake-off (first run cold-loads each model from the mirror; be patient)...")
    result, dt = call("/v1/eval", {"task": "embedding", "k": 5,
                                    "models": CANDIDATES, "dataset": eval_ds})
    # eval returns metrics keyed to k (e.g. hit@5, ndcg@5); discover them, don't hardcode.
    m0 = next((r["metrics"] for r in result["scorecard"] if "metrics" in r), {})
    ndcg_key = next((k for k in m0 if k.startswith("ndcg")), None)
    print(f"\n(bake-off wall time: {dt:.1f}s)   primary metric: {result['primary_metric'].upper()}")
    ndcg_hdr = ndcg_key.upper().replace("NDCG", "nDCG") if ndcg_key else "nDCG"
    print(f"{'model':<56}{'MRR':>7}{ndcg_hdr:>9}{'verified':>10}")
    print("-" * 84)
    for row in result["scorecard"]:
        if "error" in row:
            print(f"{row['model']:<56}  (skipped: {str(row['error'])[:34]})")
            continue
        m = row["metrics"]
        ndcg = m.get(ndcg_key, 0.0) if ndcg_key else 0.0
        print(f"{row['model']:<56}{m['mrr']:>7.3f}{ndcg:>9.3f}{str(row['verified']):>10}")
    winner = result["winner"]
    print(f"\n>>> WINNER on YOUR intents: {winner}")
    (CAP / "eval_scorecard.json").write_text(json.dumps(result, indent=2))

    # --------------------------------------------------------------------- 3. SERVE
    rule("3. SERVE  — centroids from the winner, route UNSEEN messages live")
    # Represent each intent by the CENTROID (mean vector) of its example utterances —
    # the standard production pattern for embedding-based routing. Embed all training
    # utterances once, average per intent.
    flat, owner = [], []
    for i, it in enumerate(intents):
        for utt in it["train"]:
            flat.append(utt)
            owner.append(i)
    emb, dt = call("/v1/embeddings", {"model": winner, "input": flat})
    vecs = [d["embedding"] for d in emb["data"]]
    print(f"Embedded {len(flat)} example utterances into {len(intents)} centroids in {dt*1000:.0f} ms.")
    centroids = []
    for i in range(len(intents)):
        members = [vecs[j] for j in range(len(vecs)) if owner[j] == i]
        dim = len(members[0])
        centroids.append([sum(m[d] for m in members) / len(members) for d in range(dim)])

    # Route every held-out message by nearest centroid. Embed them in ONE batched call
    # (this is also how you'd score a backlog); accuracy is what matters here.
    tmsgs = [t["message"] for t in tests]
    temb, _ = call("/v1/embeddings", {"model": winner, "input": tmsgs})
    tvecs = [d["embedding"] for d in temb["data"]]
    correct, routed = 0, []
    for t, v in zip(tests, tvecs):
        best = max(range(len(intents)), key=lambda i: cosine(v, centroids[i]))
        pred = keys[best]
        ok = (pred == t["gold"])
        correct += ok
        routed.append({"message": t["message"], "gold": t["gold"], "pred": pred, "ok": ok})
        flag = "ok " if ok else "MISS"
        print(f"  [{flag}] {t['message'][:56]:<56} -> {pred:<12} (gold {t['gold']})")
    acc = correct / len(tests)

    # Separately, measure warm PER-MESSAGE serving latency the way an agent hits it live
    # (one message per call). A small throttled probe keeps us under the demo rate limit.
    print("\nProbing warm per-message serving latency (single-message calls)...")
    lat = []
    for msg in tmsgs[:8]:
        _, d2 = call("/v1/embeddings", {"model": winner, "input": msg})
        lat.append(d2 * 1000)
    lat.sort()
    warm = lat[len(lat) // 2]
    p95 = lat[max(0, int(len(lat) * 0.95) - 1)]
    print(f"Routing accuracy on {len(tests)} UNSEEN messages: {correct}/{len(tests)} = {acc*100:.1f}%")
    print(f"Warm per-message latency (round-trip incl. network): median {warm:.0f} ms, p95 {p95:.0f} ms")

    # ------------------------------------------------------------------- 4. COMPARE
    rule("4. COMPARE  — cost at 1,000,000 routing requests / month")
    REQS = 1_000_000
    # Frontier-LLM intent call (how most agent frameworks route today): a system prompt
    # listing the intent taxonomy + the user message in, the chosen label out.
    #   ~350 input tokens (instructions + ~10 intent defs + message) and ~8 output tokens.
    #   GPT-4o-mini published pricing (2026): $0.15 / 1M input, $0.60 / 1M output.
    IN_TOK, OUT_TOK = 350, 8
    llm = (IN_TOK * 0.15 + OUT_TOK * 0.60) * REQS / 1e6
    # Frontier embedding API analog: OpenAI text-embedding-3-small, $0.02 / 1M tokens,
    # ~20 tokens per short message. (You still host the vector compare — trivial.)
    emb_api = 20 * REQS / 1e6 * 0.02
    # Bay Run winner on CPU — underlying Cloud Run compute at the MEASURED warm latency.
    #   Cloud Run (us): $0.000024/vCPU-s + $0.0000025/GiB-s; 1 vCPU + 2 GiB.
    sec = warm / 1000.0
    brun = REQS * (sec * 0.000024 + sec * 2 * 0.0000025)
    print(f"{'approach':<52}{'$ / 1M req':>12}")
    print("-" * 64)
    print(f"{'GPT-4o-mini  (LLM intent classification)':<52}{llm:>11.2f}")
    print(f"{'OpenAI text-embedding-3-small  (embed only)':<52}{emb_api:>11.2f}")
    print(f"{'Bay Run winner: '+winner.split('/')[-1]+' (CPU)':<52}{brun:>11.2f}")
    print("-" * 64)
    ratio = llm / brun if brun else float("inf")
    print(f"Bay Run intent-routing is ~{ratio:.0f}x cheaper than a GPT-4o-mini intent call,")
    print(f"at {warm:.0f} ms warm latency, {acc*100:.0f}% accuracy on unseen messages, open weights, no lock-in.")
    print("\nNo paid API was called; frontier numbers are computed from published pricing.")
    print("The Bay Run scorecard, accuracy, and latency above are all real.")

    summary = {
        "base": BASE, "winner": winner,
        "eval_primary_metric": result["primary_metric"],
        "eval_scorecard": [
            {"model": r["model"], "mrr": r.get("metrics", {}).get("mrr"),
             "verified": r.get("verified"), "error": r.get("error")}
            for r in result["scorecard"]
        ],
        "n_intents": len(intents), "n_test": len(tests),
        "routing_accuracy": acc, "routing_correct": correct,
        "warm_latency_ms_median": warm, "warm_latency_ms_p95": p95,
        "cost_per_1M": {"gpt4o_mini_llm_route": round(llm, 2),
                        "openai_embed_only": round(emb_api, 4),
                        "bay_run_cpu": round(brun, 4)},
        "cost_ratio_llm_over_bayrun": round(ratio, 1),
        "routed": routed,
    }
    (CAP / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote captured artifacts -> {CAP}/  (discover.json, eval_scorecard.json, summary.json)")


if __name__ == "__main__":
    try:
        main()
    except requests.HTTPError as e:
        print(f"\nHTTP error: {e}  body={e.response.text[:300]}", file=sys.stderr)
        sys.exit(1)
