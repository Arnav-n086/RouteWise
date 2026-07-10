# RouteWise

**Track 1 · AI Agent Track — Hybrid Token-Efficient Routing Agent**
*Smart isn't spending more. Smart is knowing when not to.*

RouteWise routes each coding query between a free local model (Ollama, on
AMD MI300X) and a paid remote model (Fireworks AI) — spending tokens only
when accuracy genuinely demands it.

---

## 1. How it works (read this first)

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
Layer 3: Local Model Attempt  ← try Ollama (codellama:7b) first, always free
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
is free. Your whole job as a team is to push as many queries as possible
into "resolved for free" while keeping accuracy above the threshold.

---

## 2. Project structure

```
routewise/
├── main.py                  entry point: interactive / single query / batch
├── requirements.txt
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

## 3. Setup — run these in order, in VS Code's terminal

```bash
# 1. Move into the project folder
cd routewise

# 2. (Recommended) create a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Pull the local model on your AMD MI300X box (needs Ollama installed)
ollama pull codellama:7b

# 5. Set your Fireworks API key
cp .env.example .env
# now edit .env and paste your real key, OR export it directly:
export FIREWORKS_API_KEY=your_key_here

# 6. Smoke-test the WHOLE pipeline with zero real API calls (safe, free, fast)
python -m eval.harness --quick --dry-run

# 7. Once Ollama + API key are real, run a real quick eval
python -m eval.harness --quick

# 8. Full baseline eval
python -m eval.harness --label baseline

# 9. Tune a threshold and compare
python -m eval.harness --threshold 6.5 --label tuned
python -m eval.harness --compare "eval/results_baseline_*.json" "eval/results_tuned_*.json"

# 10. Interactive mode (talk to the agent directly)
python main.py

# 11. Single query mode
python main.py --query "write a function to add two numbers"
```

**Tip:** step 6 (`--dry-run`) works with NO Ollama and NO API key — it uses
fake canned responses just to prove the plumbing (cache, router, verifier,
token tracker, scoring) is wired correctly. Do this FIRST, before you've
even set up Ollama, to catch bugs early.

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

- During interactive mode: type `ledger` to print it any time, or `stats`
  for the JSON summary.
- After any harness run: it auto-prints at the end.
- Programmatically: `from src.token_tracker import tracker; tracker.summary()`

Example output:
```
📒 TOKEN LEDGER
LAYER     TOKENS  QUERY / NOTE
------------------------------------------------------------
🆓 cache        0  What is a binary search tree?  served from cache
🆓 local        0  Write a hello world program...  local attempt, 77 chars
💸 remote     312  Design a distributed rate lim…  FRESH | model=deepseek-coder-v2-instruct
------------------------------------------------------------
TOTAL REMOTE TOKENS (your score): 312
```

Only 💸 remote entries count toward your score. 🆓 entries are shown so you
can see the savings your routing logic is producing — great for your demo.

---

## 6. Changes made vs. the original spec (flagged, so the team knows)

1. **`src/token_tracker.py` is a new file** — wasn't in the original design.
   Gives per-layer visibility instead of just a final total.
2. **`eval/harness.py` and `eval/test_queries.json` are new files** — the
   original doc referenced `python -m eval.harness` throughout but the file
   itself was never written. This was the actual blocker to scoring anything.
3. **`router.py`'s classifier fallback changed**: originally, any classifier
   crash defaulted straight to `"remote"` (always costs tokens). Now it falls
   back to the rule-based score instead — safer for your token budget.
4. **`verifier.py`'s syntax check changed**: originally counted colons
   (`answer.count(":") == 0`), which false-flags valid one-liners. Now uses
   Python's real `ast.parse()` on the code, which actually validates syntax.
5. **`--dry-run` mode added to the harness** so you can test/debug the whole
   pipeline before Ollama or your API key are ready.

---

## 7. Team role reminders

| Role | Owns |
|---|---|
| Local Infra | Ollama setup, MI300X, `local_model.py` latency |
| Router Logic | `router.py` rules + classifier tuning |
| Remote/API | `remote_model.py`, prompt shape, token counting |
| Eval & Metrics | `eval/harness.py`, `test_queries.json`, running comparisons |
| Cascade & Safety | `verifier.py`, `agent.py` fallback paths |
| Orchestration & Submission | README, Devpost, demo, GitHub |

---

## 8. Metrics to watch during eval

| Metric | Target |
|---|---|
| Total remote tokens | as low as possible |
| Remote call rate | < 35% |
| Routing accuracy | > 80% |
| Avg tokens per remote call | < 600 |
| Accuracy | > 80% (hard floor — don't sacrifice this for tokens) |
