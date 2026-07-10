# RouteWise

**AMD Developer Hackathon: Act II, Track 1 · AI Agent Track — Hybrid Token-Efficient Routing Agent**
*Smart isn't spending more. Smart is knowing when not to.*

RouteWise routes each coding query between a free local model (Ollama) and
a paid remote model (Fireworks AI) — spending tokens only when accuracy
genuinely demands it.

---

## 1. How it works

Every query passes through 5 layers, in order. Each layer is a chance to
answer for **zero tokens** before falling through to the next:

```
Query
  │
  ▼
Layer 1: Cache Check          ← seen this exact query before? instant, free
  │
  ▼
Layer 2: Complexity Router    ← rules + local ML classifier decide local/remote
  │
  ▼
Layer 3: Local Model Attempt  ← try Ollama (qwen3:8b) first, always free
  │
  ▼
Layer 4: Confidence Verifier  ← is local's answer actually trustworthy?
  │
  ▼
Layer 5: Targeted Remote Call ← ONLY if needed. Passes local's attempt so
                                  remote FIXES instead of redoing from scratch
  │
  ▼
Final Answer
```

**The only place real score-tokens get spent is Layer 5.** Everything else
is free!

---

## 2. Project structure

```
routewise/
├── main.py                  entry point: interactive / single query / batch
├── requirements.txt         core deps — what Docker uses
├── requirements-ml.txt      optional: adds the grey-zone ML classifier
├── Dockerfile
├── docker-compose.yml       bundles the app + an Ollama container
├── .env.example             copy to .env, fill in your Fireworks key
├── src/
│   ├── config.py             ALL tunable knobs — the only file you tune during eval
│   ├── logger.py             writes decisions to logs/routewise.log
│   ├── token_tracker.py      per-step token ledger (see section 5)
│   ├── cache.py               Layer 1
│   ├── router.py               Layer 2 (rules + HuggingFace classifier)
│   ├── local_model.py           Layer 3 (Ollama)
│   ├── verifier.py                Layer 4 (heuristic confidence check)
│   ├── remote_model.py              Layer 5 (Fireworks AI)
│   └── agent.py             orchestrates all 5 layers per query
└── eval/
    ├── test_queries.json    24 labeled queries (easy/medium/hard + 2 duplicates)
    └── harness.py            runs the test set, scores you, saves results
```

---

## 3. Setup

### Requirements
- [Docker](https://docs.docker.com/get-docker/) (recommended — no local Python/Ollama needed)
- Fireworks AI API key ([get one at fireworks.ai](https://fireworks.ai))

### Quick start (Docker)

```bash
# 1. Clone this repo
git clone https://github.com/Arnav-n086/RouteWise.git
cd RouteWise

# 2. Set your Fireworks API key
cp .env.example .env
# edit .env and paste your real key

# 3. Bring up Ollama + pull the model (first run only, ~5GB, then cached
#    in a named volume so it doesn't re-download on restarts)
docker compose up -d ollama

# 4. Run RouteWise (invoked on demand, per command — not a long-running daemon)
docker compose run --rm routewise python main.py --query "write a function to check if a number is prime"
docker compose run --rm routewise python main.py            # interactive mode
docker compose run --rm routewise python -m eval.harness --quick
```

`docker compose run` automatically starts `ollama` (and pulls the model, if
not already pulled) first — steps 3 and 4 don't need to be run in strict
sequence, that's just the clearer explanation of what's happening.

**Expected output** (real, verified — one free local answer, one that needs
a paid remote fix):

```
$ docker compose run --rm routewise python main.py --query "write a function to check if a number is prime"

{
  "served_from": "local",
  "remote_tokens_used": 0,
  "path": ["cache_miss", "complexity=0.2", "TRY_LOCAL", "VERIFY_LOCAL", "LOCAL_ACCEPTED✅"],
  "success": true
}
📒 TOKEN LEDGER
🆓 local        0  write a function to check if a number is prime
TOTAL REMOTE TOKENS (your score): 0

$ docker compose run --rm routewise python main.py --query "write a production-ready async event loop with concurrency support"

{
  "served_from": "local→remote",
  "remote_tokens_used": 1049,
  "path": ["cache_miss", "complexity=1.2", "TRY_LOCAL", "VERIFY_LOCAL", "ESCALATE→REMOTE(has_placeholders)"],
  "success": true
}
📒 TOKEN LEDGER
💸 remote    1049  write a production-ready async event loop...  FIX (cascade) | model=gpt-oss-120b
TOTAL REMOTE TOKENS (your score): 1049
```

**Docker notes:**
- `requirements.txt` is intentionally lean and excludes `transformers`/`torch`
  (only needed for the grey-zone ML classifier in `router.py`, which our own
  baseline data shows never actually triggers — the rule-based router
  resolves every test query on its own). Grey-zone queries fall back to the
  rule-based score instead. Install `requirements-ml.txt` as well (locally,
  or by adding it to the Dockerfile) if you want that path to work.
- If `docker compose up`/`pull` fails with `httpReadSeeker: failed open`,
  disable **Docker Desktop → Settings → General → "Use containerd for
  pulling and storing images"** and restart Docker Desktop — that lazy-pull
  mechanism is fragile on some networks; the classic pull mechanism isn't.

### Local setup (without Docker)

```bash
cd routewise
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
# optional, only if you want the grey-zone ML classifier in router.py to work:
pip install -r requirements-ml.txt

ollama pull qwen3:8b            # needs Ollama installed locally

cp .env.example .env            # then edit in your Fireworks key

python -m eval.harness --quick --dry-run   # smoke-test with zero real API calls
python -m eval.harness --quick             # real quick eval
python -m eval.harness --label baseline    # full baseline eval

python main.py                                          # interactive mode
python main.py --query "write a function to add two numbers"   # single query
```

**Tip:** `--dry-run` works with NO Ollama and NO API key — it uses fake
canned responses just to prove the plumbing (cache, router, verifier, token
tracker, scoring) is wired correctly. Do this FIRST to catch bugs early.

---

## 4. The 3 tuning dials (in `src/config.py`)

| Parameter | Default | Raise it → | Lower it → |
|---|---|---|---|
| `COMPLEXITY_REMOTE_THRESHOLD` | 7.5 | fewer remote calls, riskier accuracy | more remote calls, safer accuracy |
| `CONFIDENCE_THRESHOLD` | 0.72 | escalates more to remote, safer | trusts local more, cheaper |
| `REMOTE_MAX_TOKENS` | 800 | (n/a — lowering only) | fewer tokens per remote call |

**Workflow:** run `eval.harness --label baseline`, look at the summary,
adjust one dial in `config.py`, run again with a new `--label`, then
`--compare` the two. Repeat until tokens are minimized and accuracy is
still above `ACCURACY_THRESHOLD` (0.80).

---

## 5. Token tracking — how to actually see where tokens go

`src/token_tracker.py` keeps a full ledger, not just a running total.
Every cache hit, every local attempt, every remote call gets an entry.

**Interactive mode** prints a per-query report after every answer — real
routing decision, real model used, real confidence, real token cost, no
invented numbers:

```
────────────────────────────────────────────────────────────────
 Build a complete AVL tree implementation from scratch...
────────────────────────────────────────────────────────────────
 Router      RULE (matched hard phrase) -> REMOTE
 Model       accounts/fireworks/models/gpt-oss-120b (remote, paid)
 Confidence  ➡️  N/A (routed directly to remote, local never tried)
 Tokens      💸 916   (session total: 916)
 Latency     65.3s
────────────────────────────────────────────────────────────────
 ANSWER
────────────────────────────────────────────────────────────────
 (full code prints here, in full — not truncated)
────────────────────────────────────────────────────────────────
```

Type `quit` to end the session and print a SESSION SUMMARY (total queries,
cache hits, local attempts, remote calls, total tokens, avg tokens/query).
`ledger` prints the raw per-event table any time, `stats` prints the JSON
summary. Programmatically: `from src.token_tracker import tracker`.

---

## 6. Metrics to watch during eval

| Metric | Target |
|---|---|
| Total remote tokens | as low as possible |
| Remote call rate | < 35% |
| Routing accuracy | > 80% |
| Avg tokens per remote call | < 600 |
| Accuracy | > 80% (hard floor — don't sacrifice this for tokens) |

### Current baseline (24 queries, real Ollama + real Fireworks, `qwen3:8b` + `gpt-oss-120b`)

| Metric | Result |
|---|---|
| Total remote tokens | 2,113 |
| Remote call rate | 8.3% |
| Cache hit rate | 41.7% |
| Accuracy | 100% |
| Avg tokens per remote call | 1,056.5 |
| Routing accuracy | 33.3% |

Two numbers miss their stated target, both understood and left as-is rather
than blindly tuned away:

- **Avg tokens/remote call (1,056.5)** — only 2 of 24 queries escalated to
  remote, both genuinely hard (a full red-black tree, a production async
  event loop). Direct testing confirmed `gpt-oss-120b` uses the full 800
  completion-token cap on these and is still mid-answer when cut off —
  lowering `REMOTE_MAX_TOKENS` further would truncate code on exactly the
  hardest queries, risking the accuracy hard floor for a token metric. Left
  at 800 on purpose.
- **Routing accuracy (33.3%)** — this metric only checks whether
  `hard`-labeled queries were served by remote, not whether the final answer
  was correct. Several `hard`-labeled queries were answered correctly by
  local for free, which counts as a routing "miss" here despite being the
  system doing exactly what it's designed to do (try free local first, only
  pay when verification fails). Lowering `COMPLEXITY_REMOTE_THRESHOLD` to
  chase this number would mean spending tokens on queries local was already
  answering correctly — a bad trade given accuracy is already 100%.

---

## 7. Known limitations

- **Confidence verification is heuristic, not model-based.** `verifier.py`
  scores answers by string/AST pattern-matching (refusal phrases,
  placeholder markers, real Python syntax parsing, etc.), not by asking a
  model to judge quality. Edge cases exist where this could misscore — an
  unusually-phrased correct answer might trip a false-positive check, or a
  subtly wrong answer might pass every heuristic cleanly.
- **The grey-zone ML classifier's real-world accuracy is unverified.** None
  of the 24 baseline queries actually landed in the rule-based router's
  "uncertain" zone, so the HuggingFace classifier path has never been
  exercised against real traffic in this project — only confirmed to fall
  back correctly when it can't load (see `CONFIG.CLASSIFIER_TIMEOUT`). It's
  also excluded from the default Docker image (see section 3).
- **`REMOTE_MAX_TOKENS` (800) can be hit mid-answer on the hardest tasks.**
  Confirmed directly: `gpt-oss-120b` used the entire completion cap and was
  still generating on a full red-black tree implementation. Genuinely hard
  queries may come back truncated rather than complete.
- **Cache matching is exact, not semantic.** `cache.py` only normalizes by
  lowercasing and trimming whitespace — two differently-phrased but
  equivalent queries won't hit the cache and are treated as unrelated.

---

## 8. Tech stack

- Python
- [Ollama](https://ollama.com) — local inference (`qwen3:8b`)
- [Fireworks AI API](https://fireworks.ai) — remote inference (`gpt-oss-120b`)
- HuggingFace `transformers` — optional grey-zone classifier, not included
  in the default Docker image (see section 3)
- Docker / Docker Compose

---

## Built By

Arnav Nakka — Team RouteWise AMD Developer Hackathon: Act II, Track 1
