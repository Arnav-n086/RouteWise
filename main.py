#!/usr/bin/env python3
"""
main.py — RouteWise entry point.

Three modes:
  python main.py                              -> interactive REPL
  python main.py --query "write a function"   -> single query, JSON output
  python main.py --batch queries.txt          -> one query per line, batch run
"""

import sys
import argparse
import json
from src.agent import run, session_stats
from src.local_model import is_local_available
from src.config import CONFIG
from src.token_tracker import tracker
from src.logger import get_logger

logger = get_logger("main")


def _router_line(result) -> str:
    if result.served_from == "cache":
        return result.path_taken[0] if result.path_taken else "CACHE HIT (seen before)"
    profile = result.complexity_profile
    if profile is None:
        return "N/A"
    if profile.rule_decision != "uncertain":
        basis = f"RULE ({profile.reason})"
    else:
        basis = f'ML classifier ({profile.ml_confidence:.0%} confidence: "{profile.ml_label}")'
    return f"{basis} -> {profile.final_decision.upper()}"


def _model_line(result) -> str:
    served = result.served_from
    if served == "cache":
        return "— (cached answer, no model call)"
    if served == "local":
        return f"{CONFIG.LOCAL_MODEL} (local, free)"
    if served == "remote":
        return f"{CONFIG.REMOTE_MODEL} (remote, paid)"
    if served == "local→remote":
        return f"{CONFIG.LOCAL_MODEL} (local, tried first) -> {CONFIG.REMOTE_MODEL} (remote fix)"
    if served == "local_fallback":
        return f"{CONFIG.LOCAL_MODEL} (local fallback — remote call failed)"
    return served


def _confidence_line(result) -> str:
    if result.served_from == "cache":
        return "💾 N/A (cache hit)"
    v = result.verification
    if v is None:
        return "➡️  N/A (routed directly to remote, local never tried)"
    if v.is_confident:
        return f"✅ {v.confidence_score * 10:.0f}/10 — accepted, served from local"
    return f"⚠️  {v.confidence_score * 10:.0f}/10 — escalated ({', '.join(v.failures)})"


_WIDTH = 64


def print_query_report(result):
    """Per-query report using only real, measured fields — no invented $ costs.

    Metadata (routing/model/confidence/tokens) is kept compact up top so the
    ANSWER section — the part anyone actually reads — is visually unmissable
    below its own header, not buried inline as just another labeled field.
    """
    tokens_tag = "🆓" if result.remote_tokens_used == 0 else "💸"

    print("\n" + "─" * _WIDTH)
    print(f" {result.query}")
    print("─" * _WIDTH)
    print(f" Router      {_router_line(result)}")
    print(f" Model       {_model_line(result)}")
    print(f" Confidence  {_confidence_line(result)}")
    print(f" Tokens      {tokens_tag} {result.remote_tokens_used}   (session total: {tracker.total_remote_tokens})")
    print(f" Latency     {result.total_latency:.1f}s")
    print("─" * _WIDTH)
    print(" ANSWER")
    print("─" * _WIDTH)
    print(result.answer)
    print("─" * _WIDTH)


def startup_checks():
    print("\n🔍 RouteWise — Startup Checks")
    print("─" * 40)
    local_ok = is_local_available()
    print(f"  Local model ({CONFIG.LOCAL_MODEL}): {'✅' if local_ok else '❌'}")
    api_ok = bool(CONFIG.FIREWORKS_API_KEY)
    print(f"  Fireworks API key: {'✅' if api_ok else '❌ set FIREWORKS_API_KEY'}")
    print(f"  HuggingFace classifier: ✅ (loads on first grey-zone query)")
    if not api_ok:
        print("\n❌ Fix: export FIREWORKS_API_KEY=your_key_here\n")
        sys.exit(1)
    print("─" * 40)
    print("✅ Ready\n")
    return local_ok and api_ok


def run_interactive():
    print("\n🚀 RouteWise Interactive Mode")
    print("   Commands: 'stats', 'ledger', 'clear cache', 'quit'\n")
    while True:
        try:
            query = input("Query> ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not query:
            continue
        if query.lower() == "quit":
            tracker.print_summary()
            break
        if query.lower() == "stats":
            print(json.dumps(session_stats(), indent=2))
            continue
        if query.lower() == "ledger":
            tracker.print_ledger()
            continue
        if query.lower() == "clear cache":
            from src.cache import cache
            cache.clear()
            print("Cache cleared.\n")
            continue

        result = run(query)
        print_query_report(result)


def run_single_query(query: str):
    result = run(query)
    print(json.dumps({
        "query": result.query,
        "answer": result.answer,
        "remote_tokens_used": result.remote_tokens_used,
        "served_from": result.served_from,
        "path": result.path_taken,
        "latency": result.total_latency,
        "success": result.success,
    }, indent=2))
    tracker.print_ledger(last_n=1)


def run_batch(filepath: str):
    with open(filepath) as f:
        queries = [line.strip() for line in f if line.strip()]
    total_tokens = 0
    for i, query in enumerate(queries, 1):
        result = run(query)
        total_tokens += result.remote_tokens_used
        print(json.dumps({
            "id": i, "query": query, "answer": result.answer,
            "remote_tokens": result.remote_tokens_used,
            "served_from": result.served_from,
        }))
    print(f"\n✅ Done. Total remote tokens: {total_tokens}", file=sys.stderr)
    tracker.print_ledger()
    tracker.print_summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RouteWise")
    parser.add_argument("--query", "-q", type=str)
    parser.add_argument("--batch", "-b", type=str)
    parser.add_argument("--skip-checks", action="store_true")
    args = parser.parse_args()

    if not args.skip_checks:
        startup_checks()

    if args.query:
        run_single_query(args.query)
    elif args.batch:
        run_batch(args.batch)
    else:
        run_interactive()
