#!/usr/bin/env python3
"""
eval/harness.py — 🆕 NEW FILE (this was referenced everywhere in the spec
but never actually written — this is the missing piece).

WHAT IT DOES:
  1. Loads eval/test_queries.json
  2. Runs every query through the full agent (all 5 layers)
  3. Computes: total remote tokens, remote call rate, cache hit rate,
     accuracy (keyword match), routing accuracy (vs expected_difficulty)
  4. Saves a timestamped results JSON to eval/
  5. Prints a token ledger + summary table

USAGE:
  python -m eval.harness --quick                 # first 8 queries only, fast smoke test
  python -m eval.harness --label baseline         # full run, saved as "baseline"
  python -m eval.harness --threshold 6.5 --label tuned   # override a dial for this run
  python -m eval.harness --compare eval/results_baseline_*.json eval/results_tuned_*.json
  python -m eval.harness --dry-run                # NO real API/Ollama calls — uses
                                                    # fake canned answers so you can test
                                                    # the harness plumbing itself before
                                                    # you have keys/Ollama set up.
"""

import argparse
import glob
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CONFIG
from src.token_tracker import tracker
from src.logger import get_logger

logger = get_logger("harness")

TEST_QUERIES_PATH = Path(__file__).parent / "test_queries.json"
RESULTS_DIR = Path(__file__).parent


def load_test_queries(quick: bool = False) -> list[dict]:
    with open(TEST_QUERIES_PATH) as f:
        queries = json.load(f)
    if quick:
        queries = queries[:8]
    return queries


def apply_dry_run_mocks():
    """
    Patches call_local / call_remote with fake, deterministic responses so
    the ENTIRE pipeline (cache, router, verifier, agent, harness scoring)
    can be exercised with zero real API calls and no Ollama required.

    This is for testing the harness itself — NOT for producing your real
    submission score. Remove --dry-run once your API key + Ollama are ready.
    """
    import src.local_model as local_model
    import src.remote_model as remote_model

    def fake_call_local(query):
        # Simulate: local does fine on short/easy-looking queries,
        # struggles (short/placeholder answer) on longer ones.
        if len(query) < 80:
            return f"def solution():\n    # handles: {query[:30]}\n    return True", 0.4
        return "# TODO: not sure how to do this", 0.4

    def fake_call_remote(query, local_attempt=None):
        fake_tokens = 150 if local_attempt else 300
        answer = f"def solution():\n    # remote fix for: {query[:40]}\n    return True"
        remote_model._total_remote_tokens += fake_tokens
        remote_model._total_remote_calls += 1
        tracker.record(query, "remote", fake_tokens, note="DRY-RUN mock call")
        return answer, fake_tokens, 0.2

    local_model.call_local = fake_call_local
    remote_model.call_remote = fake_call_remote

    # agent.py imported these by name, so patch its references too
    import src.agent as agent
    agent.call_local = fake_call_local
    agent.call_remote = fake_call_remote


def check_accuracy(answer: str, expected: list[str]) -> bool:
    if not expected:
        # Hard queries with no keyword list — accuracy just means "got a
        # non-trivial, non-empty answer"
        return bool(answer) and len(answer.strip()) > 30
    a_lower = answer.lower()
    return all(kw.lower() in a_lower for kw in expected)


def run_eval(queries: list[dict], threshold_override: float = None) -> dict:
    if threshold_override is not None:
        CONFIG.COMPLEXITY_REMOTE_THRESHOLD = threshold_override
        logger.info(f"Overriding COMPLEXITY_REMOTE_THRESHOLD -> {threshold_override}")

    from src.agent import run as agent_run
    from src.cache import cache

    per_query_results = []
    routing_correct = 0
    routing_total = 0
    accuracy_hits = 0

    for item in queries:
        query = item["query"]
        expected_difficulty = item.get("expected_difficulty")
        expected_kw = item.get("answer_must_contain", [])

        result = agent_run(query)

        acc = check_accuracy(result.answer, expected_kw)
        if acc:
            accuracy_hits += 1

        routing_match = None
        if expected_difficulty in ("easy", "hard") and result.served_from != "cache":
            routing_total += 1
            predicted_local = result.served_from in ("local",)
            predicted_remote = "remote" in result.served_from
            if expected_difficulty == "easy" and predicted_local:
                routing_correct += 1
                routing_match = True
            elif expected_difficulty == "hard" and predicted_remote:
                routing_correct += 1
                routing_match = True
            else:
                routing_match = False

        per_query_results.append({
            "id": item["id"],
            "query": query,
            "expected_difficulty": expected_difficulty,
            "served_from": result.served_from,
            "remote_tokens": result.remote_tokens_used,
            "accuracy_pass": acc,
            "routing_match": routing_match,
            "path": result.path_taken,
        })

    n = len(queries)
    summary = {
        "label": None,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config_snapshot": {
            "COMPLEXITY_REMOTE_THRESHOLD": CONFIG.COMPLEXITY_REMOTE_THRESHOLD,
            "CONFIDENCE_THRESHOLD": CONFIG.CONFIDENCE_THRESHOLD,
            "REMOTE_MAX_TOKENS": CONFIG.REMOTE_MAX_TOKENS,
        },
        "total_queries": n,
        "total_remote_tokens": tracker.total_remote_tokens,
        "remote_call_rate": round(sum(1 for r in per_query_results if "remote" in r["served_from"]) / n, 3),
        "cache_hit_rate": cache.stats()["hit_rate"],
        "avg_tokens_per_remote_call": round(
            tracker.total_remote_tokens / max(sum(1 for r in per_query_results if "remote" in r["served_from"]), 1), 1
        ),
        "accuracy": round(accuracy_hits / n, 3),
        "accuracy_pass_threshold": accuracy_hits / n >= CONFIG.ACCURACY_THRESHOLD,
        "routing_accuracy": round(routing_correct / max(routing_total, 1), 3) if routing_total else None,
        "per_query": per_query_results,
    }
    return summary


def print_summary(summary: dict):
    print("\n" + "=" * 60)
    print("📊 EVAL SUMMARY")
    print("=" * 60)
    print(f"  Queries run:              {summary['total_queries']}")
    print(f"  Total remote tokens:      {summary['total_remote_tokens']}   <-- this is your score")
    print(f"  Remote call rate:         {summary['remote_call_rate']*100:.1f}%  (target < 35%)")
    print(f"  Cache hit rate:           {summary['cache_hit_rate']}")
    print(f"  Avg tokens/remote call:   {summary['avg_tokens_per_remote_call']}  (target < 600)")
    print(f"  Accuracy:                 {summary['accuracy']*100:.1f}%  (target > {CONFIG.ACCURACY_THRESHOLD*100:.0f}%)")
    print(f"  Accuracy passes threshold:{' ✅' if summary['accuracy_pass_threshold'] else ' ❌ FIX THIS FIRST'}")
    if summary["routing_accuracy"] is not None:
        print(f"  Routing accuracy:         {summary['routing_accuracy']*100:.1f}%  (target > 80%)")
    print("=" * 60)
    tracker.print_ledger()


def save_results(summary: dict, label: str) -> str:
    summary["label"] = label
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"results_{label}_{ts}.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n💾 Results saved to: {path}")
    return str(path)


def compare_results(paths: list[str]):
    expanded = []
    for p in paths:
        expanded.extend(glob.glob(p))
    if len(expanded) < 2:
        print("Need at least 2 result files to compare.")
        return

    rows = []
    for p in expanded:
        with open(p) as f:
            data = json.load(f)
        rows.append(data)

    print("\n" + "=" * 90)
    print(f"{'LABEL':<15}{'REMOTE TOK':<12}{'CALL RATE':<12}{'ACCURACY':<12}{'ROUTING ACC':<12}{'THRESHOLD':<10}")
    print("-" * 90)
    for r in rows:
        thr = r.get("config_snapshot", {}).get("COMPLEXITY_REMOTE_THRESHOLD", "?")
        routing = r.get("routing_accuracy")
        routing_str = f"{routing*100:.1f}%" if routing is not None else "n/a"
        print(f"{r.get('label','?'):<15}{r['total_remote_tokens']:<12}{r['remote_call_rate']*100:.1f}%{'':<6}"
              f"{r['accuracy']*100:.1f}%{'':<6}{routing_str:<12}{thr:<10}")
    print("=" * 90)
    best = min(rows, key=lambda r: r["total_remote_tokens"] if r["accuracy_pass_threshold"] else float("inf"))
    print(f"\n🏆 Best (lowest tokens while passing accuracy): {best.get('label')} "
          f"({best['total_remote_tokens']} tokens)\n")


def main():
    parser = argparse.ArgumentParser(description="RouteWise eval harness")
    parser.add_argument("--quick", action="store_true", help="run only first 8 test queries")
    parser.add_argument("--label", type=str, default="run", help="name for this eval run")
    parser.add_argument("--threshold", type=float, default=None, help="override COMPLEXITY_REMOTE_THRESHOLD")
    parser.add_argument("--compare", nargs="+", help="compare 2+ results_*.json files")
    parser.add_argument("--dry-run", action="store_true", help="use fake mock model calls (no real API/Ollama)")
    args = parser.parse_args()

    if args.compare:
        compare_results(args.compare)
        return

    if args.dry_run:
        print("⚠️  DRY-RUN MODE: using fake mock responses, NOT real Fireworks/Ollama calls.\n")
        apply_dry_run_mocks()

    tracker.reset()
    queries = load_test_queries(quick=args.quick)
    print(f"Running eval on {len(queries)} queries" + (" (quick mode)" if args.quick else "") + "...\n")

    summary = run_eval(queries, threshold_override=args.threshold)
    print_summary(summary)
    save_results(summary, args.label)


if __name__ == "__main__":
    main()
