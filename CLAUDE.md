# RouteWise — Project Context for Claude Code

Solo hackathon project (Track 1, AI Agent Track). First coding project for
the author — be a mentor, not just an executor: explain what you're doing
and why as you go, don't just make changes silently.

## What this is

A token-efficient routing agent. Every coding query goes through 5 layers
(cache -> rule/ML router -> local Ollama model -> confidence verifier ->
targeted Fireworks AI remote fallback). Full architecture is in `README.md`
— read that first, it's the source of truth for design decisions.

## Current status

- All core files are written and unit-verified via `eval/harness.py --dry-run`
  (fake mock models, no real API/Ollama needed) — confirmed working.
- NOT yet tested against real Ollama or a real Fireworks API key.
- No tuning has happened yet — all 3 dials in `src/config.py` are still
  at their original defaults.

## What's next (in order)

1. Confirm Ollama is installed and reachable, `ollama pull codellama:7b`
2. Set `FIREWORKS_API_KEY` (see `.env.example`)
3. Run `python -m eval.harness --quick` (drop `--dry-run` now that infra is real)
4. Look at real token numbers, tune `src/config.py`'s 3 dials, re-run, compare
5. Only after tokens/accuracy look reasonable: write README polish + demo

## Rules for working in this repo

- `src/config.py` is the ONLY file that should change during tuning — don't
  hardcode thresholds elsewhere.
- Every change to routing/verification logic should be flagged and explained
  in chat before being applied, not silently edited — this is a learning
  project, not a black box.
- Don't reintroduce dependencies on transformers/torch unless the grey-zone
  ML classifier path is actually being exercised — it's designed to
  gracefully degrade without them.
