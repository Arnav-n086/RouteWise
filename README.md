# RouteWise

**AMD Developer Hackathon: Act II, Track 1 · AI Agent Track — Hybrid Token-Efficient Routing Agent**
*Smart isn't spending more. Smart is knowing when not to.*

RouteWise routes each coding query between a free local model (Ollama) and
a paid remote model (Fireworks AI) — spending tokens only when accuracy
genuinely demands it.

## Demo — the headline numbers

Real run, not simulated: 50 labeled queries (16 easy, 16 medium, 12
genuinely-hard, 6 duplicates) through the full 5-layer pipeline against live
Ollama (`qwen3:8b`) + live Fireworks (`gpt-oss-120b`). Full breakdown,
per-query table, and methodology in section 6.

| | |
|---|---|
| Queries resolved for **$0** | 36 / 50 (72%) |
| Answer accuracy | 100% |
| Routing accuracy (right model for the query) | 96.4% |
| Total remote tokens across all 50 queries | 18,235 |

Try it yourself:
```bash
python main.py --query "write a function to check if a number is prime"
# -> served locally, 0 tokens

python main.py --query "design a distributed rate limiter with full implementation"
# -> routed straight to remote, ~1600 tokens — only paid when it's genuinely earned
```

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
│   ├── cache.py               Layer 1 (exact + optional semantic matching)
│   ├── router.py               Layer 2 (rules + HuggingFace classifier)
│   ├── local_model.py           Layer 3 (Ollama)
│   ├── verifier.py                Layer 4 (heuristic confidence check)
│   ├── remote_model.py              Layer 5 (Fireworks AI)
│   └── agent.py             orchestrates all 5 layers per query
└── eval/
    ├── test_queries.json    50 labeled queries (easy/medium/hard + 6 duplicates)
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
  "served_from": "remote",
  "remote_tokens_used": 1602,
  "path": ["cache_miss", "complexity=0.3", "SKIP_LOCAL→REMOTE"],
  "success": true
}
📒 TOKEN LEDGER
💸 remote    1602  write a production-ready async event loop...  FRESH | model=gpt-oss-120b
TOTAL REMOTE TOKENS (your score): 1602
```

("production-ready" and "async event loop" are both `HARD_PHRASES` — the
router skips local entirely and goes straight to remote, rather than trying
local first and cascading. See section 1.)

**Docker notes:**
- `requirements.txt` is intentionally lean and excludes the ML extras
  (`requirements-ml.txt`): the grey-zone classifier in `router.py` (our own
  baseline data shows it never actually triggers — the rule-based router
  resolves every test query on its own) and semantic cache matching in
  `cache.py` (catches differently-phrased-but-equivalent queries, e.g.
  "sum two numbers" vs "add two numbers"). Both fall back gracefully to
  their non-ML behavior without it. Install `requirements-ml.txt` as well
  (locally, or by adding it to the Dockerfile) if you want either to work.
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
| `REMOTE_MAX_TOKENS` | 1500 | less truncation on hard "from scratch" tasks, more tokens | fewer tokens, more truncation risk |

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
 Tokens      💸 1595   (session total: 1595)
 Latency     15.7s
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

### Current baseline (50 queries, real Ollama + real Fireworks, `qwen3:8b` + `gpt-oss-120b`)

| Metric | Result |
|---|---|
| Total remote tokens | 18,235 |
| Remote call rate | 28.0% |
| Cache hit rate | 12.0% |
| Accuracy | 100% |
| Avg tokens per remote call | 1,302.5 |
| Routing accuracy | 96.4% |

This 50-query set is deliberately harder than a real day-to-day mix — 12 of
the 50 queries were authored to be genuinely hard on purpose (all
`HARD_PHRASES` matches, all correctly routed straight to remote). One number
still misses its stated target, understood as a deliberate tradeoff rather
than something to keep chasing:

- **Avg tokens/remote call (1,302.5)** — see the `REMOTE_MAX_TOKENS` writeup
  in section 7. Short version: direct testing showed every hard query was
  being truncated mid-code at the previous 800 cap (`finish_reason:
  "length"`, one case cut off mid-statement). Raised to 1500 as a deliberate
  middle ground — fixes 4 of 12 hard queries outright, leaves the other 8
  still truncated, at a real +51% token cost. Full coverage would need
  ~2500+ and roughly double the total again, for two queries ("design a
  full microservice architecture", "distributed message queue with
  sharding+replication") that are genuinely asking for an entire system in
  one call. Not chasing the `<600` target further on purpose.

**Router fix validated by this run — hard-phrase false positives on
definitional questions.** An earlier version of this baseline caught
`router.py`'s `HARD_PHRASES` check firing *before* the `EASY_PHRASES`/length
check, so "What is a neural network?" and "Explain what backpropagation is."
(#9, #10) were routing to remote and costing ~970 tokens despite being
trivial to answer. Fixed in `router.py` — a hard-phrase match now only wins
outright when the query *isn't* also a short, low-complexity, easy-phrased
question (see the code comment on `_compute_complexity` for the exact
guard). Verified two ways: an offline check confirmed all 50 queries still
route correctly with zero regressions (including #37, a genuinely hard
Dijkstra's-algorithm query that also happens to contain "explain" — it
correctly stays on remote because its complexity score is high), and a live
re-run showed #9/#10 resolving locally for free, which is why routing
accuracy jumped from 89.3% to 96.4%.

### Full per-query results (all 50 baseline queries)

Real output from `eval.harness --label tuned1500` (2026-07-11), not
summarized or cherry-picked — every query that ran, in order, including the
6 intentional duplicates (#45–#50) used to test the cache.

| # | Query | Difficulty | Served from | Tokens | Accuracy | Notes |
|---|---|---|---|---|---|---|
| 1 | What is a binary search tree? | easy | 🆓 local | 0 | ✅ | complexity 0.9 |
| 2 | Explain what recursion means in programming. | easy | 🆓 local | 0 | ✅ | complexity 1.0 |
| 3 | Write a Python function to add two numbers. | easy | 🆓 local | 0 | ✅ | complexity 0.2 |
| 4 | Fix this syntax error: `def foo(x)\n  return x+1` | easy | 💸 local→remote | 482 | ✅ | escalated: `broken_python_syntax` |
| 5 | What's the difference between a list and a tuple in Python? | easy | 🆓 local | 0 | ✅ | complexity 1.0 |
| 6 | Write a hello world program in Python. | easy | 🆓 local | 0 | ✅ | complexity 0.2 |
| 7 | Define what a hash map is. | easy | 🆓 local | 0 | ✅ | complexity 0.1 |
| 8 | Write a function to reverse a string in Python. | easy | 🆓 local | 0 | ✅ | complexity 0.2 |
| 9 | What is a neural network? | easy | 🆓 local | 0 | ✅ | previously a hard-phrase false positive ("neural network") — fixed, see note above |
| 10 | Explain what backpropagation is. | easy | 🆓 local | 0 | ✅ | previously a hard-phrase false positive ("backpropagation") — fixed, see note above |
| 11 | What is the difference between HTTP and HTTPS? | easy | 🆓 local | 0 | ✅ | complexity 0.9 |
| 12 | What is a variable in programming? | easy | 🆓 local | 0 | ✅ | complexity 0.2 |
| 13 | What does a compiler do? | easy | 🆓 local | 0 | ✅ | complexity 0.1 |
| 14 | Write a simple SQL query to select all rows from a table named users. | easy | 🆓 local | 0 | ✅ | complexity 0.3 |
| 15 | Explain what Big-O notation is. | easy | 🆓 local | 0 | ✅ | complexity 0.1 |
| 16 | What is the difference between a stack and a queue? | easy | 🆓 local | 0 | ✅ | complexity 0.9 |
| 17 | Write a Python function that checks if a number is prime, and also returns its smallest prime factor. | medium | 🆓 local | 0 | ✅ | complexity 1.9 |
| 18 | Implement a function to find the second largest element in a list, handling duplicates. | medium | 🆓 local | 0 | ✅ | complexity 0.4 |
| 19 | Write a Python class representing a simple stack with push, pop, and peek methods. | medium | 🆓 local | 0 | ✅ | complexity 1.1 |
| 20 | Implement binary search on a sorted list using recursion. | medium | 🆓 local | 0 | ✅ | complexity 1.1 |
| 21 | Write a function that merges two sorted lists into one sorted list. | medium | 🆓 local | 0 | ✅ | complexity 0.3 |
| 22 | Implement a basic LRU cache using a dictionary and a doubly linked list. | medium | 🆓 local | 0 | ✅ | complexity 1.1 |
| 23 | Write a function to compute the factorial of a number using memoization. | medium | 🆓 local | 0 | ✅ | complexity 1.2 |
| 24 | Explain what a REST API is and how it differs from GraphQL. | medium | 🆓 local | 0 | ✅ | complexity 2.6 |
| 25 | Write a JavaScript function to debounce another function. | medium | 💸 local→remote | 484 | ✅ | escalated: `has_placeholders` |
| 26 | Write a Java method that checks if a string is a palindrome. | medium | 🆓 local | 0 | ✅ | complexity 0.3 |
| 27 | Explain the difference between synchronous and asynchronous programming. | medium | 🆓 local | 0 | ✅ | complexity 1.1 |
| 28 | Write a Bash script that counts the number of lines in a file. | medium | 🆓 local | 0 | ✅ | complexity 0.3 |
| 29 | Implement a function to check if two strings are anagrams of each other. | medium | 🆓 local | 0 | ✅ | complexity 0.4 |
| 30 | Explain what dependency injection is and why it's used. | medium | 🆓 local | 0 | ✅ | complexity 1.0 |
| 31 | Write a function to flatten a nested list in Python. | medium | 🆓 local | 0 | ✅ | complexity 0.3 |
| 32 | Explain the difference between a process and a thread. | medium | 🆓 local | 0 | ✅ | complexity 1.0 |
| 33 | Design a distributed rate limiter that works across multiple microservices, with full implementation. | hard | 💸 remote | 1,296 | ✅ | complexity 3.6, finished under the 1500 cap |
| 34 | Implement a compiler front-end that tokenizes and parses a simple arithmetic expression grammar from scratch. | hard | 💸 remote | 1,609 | ✅ | complexity 2.8, still truncated at 1500 cap |
| 35 | Write a production-ready async event loop with concurrency support and full error handling. | hard | 💸 remote | 1,606 | ✅ | complexity 1.2, still truncated at 1500 cap |
| 36 | Design a scalable, distributed message queue architecture with sharding and replication, end to end. | hard | 💸 remote | 1,616 | ✅ | complexity 4.3, still truncated (needs 2500+) |
| 37 | Implement Dijkstra's algorithm with a full working example, then explain time complexity, then also add a priority queue optimization. | hard | 💸 remote | 944 | ✅ | complexity 3.6, finished under the 1500 cap |
| 38 | Build a complete red-black tree implementation from scratch with insert, delete, and rebalancing. | hard | 💸 remote | 1,610 | ✅ | complexity 2.7, still truncated at 1500 cap |
| 39 | Design a microservice architecture for an e-commerce platform, including database schema and message queue setup. | hard | 💸 remote | 1,617 | ✅ | complexity 2.9, still truncated (needs 2500+) |
| 40 | Implement a basic garbage collector for a toy language, handling memory management and reference counting. | hard | 💸 remote | 1,608 | ✅ | complexity 1.2, still truncated at 1500 cap |
| 41 | Write a multithreading-safe producer-consumer queue implementation with proper concurrency controls. | hard | 💸 remote | 1,072 | ✅ | complexity 1.3, finished under the 1500 cap |
| 42 | Design and implement a simple neural network from scratch, including backpropagation, without using any ML libraries. | hard | 💸 remote | 1,075 | ✅ | complexity 3.6, finished under the 1500 cap |
| 43 | Implement a B-tree data structure from scratch, supporting insert, delete, and search operations. | hard | 💸 remote | 1,609 | ✅ | complexity 2.7, still truncated at 1500 cap |
| 44 | Write a full implementation of a regex engine that supports basic pattern matching from scratch. | hard | 💸 remote | 1,607 | ✅ | complexity 2.8, still truncated at 1500 cap |
| 45 | What is a binary search tree? *(duplicate of #1)* | easy | 💾 cache | 0 | ✅ | |
| 46 | Write a Python function to add two numbers. *(duplicate of #3)* | easy | 💾 cache | 0 | ✅ | |
| 47 | What is a neural network? *(duplicate of #9)* | easy | 💾 cache | 0 | ✅ | proves cache makes a would-be-remote query free on repeat |
| 48 | Write a Python function that checks if a number is prime... *(duplicate of #17)* | medium | 💾 cache | 0 | ✅ | |
| 49 | Design a distributed rate limiter... *(duplicate of #33)* | hard | 💾 cache | 0 | ✅ | proves cache makes a hard, remote-routed query free on repeat |
| 50 | Build a complete red-black tree... *(duplicate of #38)* | hard | 💾 cache | 0 | ✅ | proves cache makes a hard, remote-routed query free on repeat |

**36 of 50 queries (72%) resolved for zero tokens.** Lower than the old
24-query baseline's 91.7% by design — this set adds 12 genuinely hard queries
(all correctly routed to remote, all answered accurately). #9 and #10 now
join the free-tier after the router fix above. Every duplicate query
(#45–#50), including the two hard ones that originally cost tokens, came
back free from cache on repeat — direct proof `CASCADE_ENABLED` and Layer 1
caching both work as designed.

---

## 7. Known limitations

- **Confidence verification is heuristic, not model-based.** `verifier.py`
  pattern-matches (refusal phrases, placeholders, real AST parsing), it
  doesn't ask a model to judge quality — an unusual correct answer could
  trip a false positive, or a subtly wrong one could pass cleanly.
- ~~`HARD_PHRASES` can misfire on simple definitional questions~~ **Fixed.**
  `router.py` was checking hard-phrase matches before easy-phrase/length
  checks, so "What is a neural network?" routed straight to remote despite
  being trivial to answer — confirmed costing ~970 tokens across 2 queries
  in the 50-query baseline (section 6). Now a hard-phrase match only wins
  outright when the query isn't also short, easy-phrased, and independently
  low-complexity; verified against all 50 baseline queries with zero
  regressions (including a genuinely hard query that also contains
  "explain").
- **The grey-zone ML classifier remains unexercised.** Across 50 baseline
  queries, none landed in the rule router's true "uncertain" zone — every
  query resolved via a phrase match or a decisive complexity score. Falls
  back cleanly on load failure (`CONFIG.CLASSIFIER_TIMEOUT`), but its
  real-world accuracy is still unproven. Excluded from the default Docker
  image (section 3).
- **`REMOTE_MAX_TOKENS` is a deliberate, incomplete tradeoff, not a solved
  problem.** Direct testing (`finish_reason` inspection, not guesswork)
  showed every one of the 12 hard-labeled queries was being truncated
  mid-code at the original 800 cap — one case cut off mid-statement,
  returning syntactically broken code. Raised to 1500: fixes 4 of 12
  outright, for a real +51% token cost across the baseline. The other 8
  still truncate. Full coverage would need ~2500+ and roughly double total
  tokens again — and even then, 2 queries ("design a full microservice
  architecture", "distributed message queue with sharding+replication")
  didn't finish at a 2500-token probe, since they're genuinely asking for an
  entire system in one call. Chose the bounded middle ground over chasing
  full coverage, given the project is scored on token efficiency. Also
  worth noting: `eval/harness.py`'s `check_accuracy()` only checks
  "non-empty, >30 chars" for hard queries with no keyword list — it doesn't
  detect truncation or invalid syntax, so the 100% accuracy figure doesn't
  catch this on its own.
- **HuggingFace downloads are intermittent, and the semantic-cache
  threshold (`CACHE_SIMILARITY_THRESHOLD = 0.90`) is unverified.** The
  embedding model loaded fine during the 50-query run but timed out in two
  smaller runs on the same machine — no root cause found. Degrades cleanly
  to exact-match-only on failure. Since every duplicate in the test set is
  an exact string repeat, the 0.90 threshold has never actually been
  exercised against differently-phrased near-duplicates.

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
