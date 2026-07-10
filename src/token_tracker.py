"""
token_tracker.py — 🆕 NEW MODULE (not in the original spec).

WHY THIS EXISTS:
The original design only counted tokens inside remote_model.py, as a running
total. That tells you the final number but not the STORY — which queries
cost what, and how much you saved by routing locally instead.

This module keeps a full ledger: one entry per layer-decision, for every
query. You can print it after every query, or dump the whole session at
the end. Only REMOTE entries count toward your actual score — everything
else (cache/local) is shown so you can see the savings, not just the cost.
"""

from dataclasses import dataclass, field
from typing import Optional
from src.logger import get_logger

logger = get_logger("token_tracker")


@dataclass
class TokenEvent:
    query_preview: str      # first ~50 chars of the query, for readability
    layer: str               # "cache" | "local" | "remote"
    tokens: int               # tokens actually billed (0 for cache/local)
    note: str = ""


class TokenTracker:
    def __init__(self):
        self.events: list[TokenEvent] = []

    def record(self, query: str, layer: str, tokens: int, note: str = ""):
        event = TokenEvent(
            query_preview=(query[:50] + "…") if len(query) > 50 else query,
            layer=layer,
            tokens=tokens,
            note=note,
        )
        self.events.append(event)
        tag = "💸" if layer == "remote" else "🆓"
        logger.info(f"{tag} TOKEN EVENT [{layer.upper()}] tokens={tokens} | {note}")

    @property
    def total_remote_tokens(self) -> int:
        return sum(e.tokens for e in self.events if e.layer == "remote")

    @property
    def total_queries(self) -> int:
        # a "query" is any event chain; we count cache hits as their own query too
        return len(self.events)

    def breakdown(self) -> dict:
        """Count of queries served by each layer, plus token cost per layer."""
        counts = {"cache": 0, "local": 0, "remote": 0}
        tokens = {"cache": 0, "local": 0, "remote": 0}
        for e in self.events:
            counts[e.layer] = counts.get(e.layer, 0) + 1
            tokens[e.layer] = tokens.get(e.layer, 0) + e.tokens
        return {"counts": counts, "tokens": tokens}

    def print_ledger(self, last_n: Optional[int] = None):
        """Human-readable table of recent token events."""
        rows = self.events[-last_n:] if last_n else self.events
        print("\n📒 TOKEN LEDGER")
        print(f"{'LAYER':<8} {'TOKENS':>7}  QUERY / NOTE")
        print("-" * 60)
        for e in rows:
            tag = "💸" if e.layer == "remote" else "🆓"
            print(f"{tag} {e.layer:<6} {e.tokens:>7}  {e.query_preview}  {e.note}")
        print("-" * 60)
        print(f"TOTAL REMOTE TOKENS (your score): {self.total_remote_tokens}\n")

    def summary(self) -> dict:
        b = self.breakdown()
        return {
            "total_remote_tokens": self.total_remote_tokens,
            "total_queries": self.total_queries,
            "served_by_layer_counts": b["counts"],
            "tokens_spent_by_layer": b["tokens"],
        }

    def reset(self):
        self.events = []


# Single shared instance — every module imports THIS object, same pattern as CONFIG.
tracker = TokenTracker()
