"""
router.py — Layer 2: decides local vs remote BEFORE any model sees the query.

TWO STAGES:
  Stage 1 — rule-based pre-filter. Instant, zero cost. Catches obvious cases.
  Stage 2 — HuggingFace zero-shot classifier, ONLY for queries rules can't
            confidently call ("the grey zone"). Still runs locally = zero
            Fireworks tokens spent on the routing decision itself.

Both stages run for free. The only thing that costs tokens is what happens
AFTER routing (agent.py Layer 5).
"""

import threading
from dataclasses import dataclass
from typing import Optional
from src.config import CONFIG
from src.logger import get_logger

logger = get_logger("router")

_classifier = None  # lazy-loaded so we don't pay model load time unless needed


def get_classifier():
    """Loads the HuggingFace zero-shot classifier once, reuses it after that."""
    global _classifier
    if _classifier is None:
        logger.info("Loading HuggingFace classifier (first grey-zone query)...")
        from transformers import pipeline
        _classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=-1,  # CPU — safe default across machines. If your MI300X
                        # box has a working GPU pipeline for transformers,
                        # you can try device=0, but benchmark latency first.
        )
        logger.info("✅ Classifier loaded")
    return _classifier


@dataclass
class ComplexityProfile:
    query_length: int
    requirement_count: int
    deep_tech_signals: int
    abstraction_signals: int
    expects_long_output: bool
    total_score: float
    rule_decision: str
    ml_confidence: Optional[float] = None
    ml_label: Optional[str] = None
    final_decision: str = "local"
    reason: str = ""


HARD_PHRASES = [
    "from scratch", "distributed system", "microservice",
    "implement a compiler", "write a parser", "memory management",
    "concurrency", "multithreading", "async event loop",
    "design a system", "production ready", "scalable architecture",
    "full implementation", "end to end", "step by step implementation",
    "dynamic programming", "dijkstra", "red-black tree", "b-tree",
    "neural network", "machine learning model", "backpropagation",
]

EASY_PHRASES = [
    "what is", "explain", "what does", "how does",
    "fix this syntax", "correct this", "why is this error",
    "simple example", "basic example", "hello world",
    "what's the difference between", "define",
]

DEEP_TECH_TERMS = [
    "recursion", "memoization", "dynamic programming",
    "binary search tree", "avl tree", "heap", "trie",
    "graph traversal", "bfs", "dfs", "topological sort",
    "dijkstra", "bellman-ford", "threading", "mutex",
    "garbage collection", "memory leak", "pointer arithmetic",
    "regex engine", "tokenizer", "ast", "abstract syntax tree",
]

ABSTRACTION_TERMS = [
    "design", "architecture", "scalable", "distributed",
    "microservice", "event driven", "message queue",
    "rest api", "graphql", "websocket", "oauth",
    "database schema", "normalization", "sharding", "replication",
]


def _compute_complexity(query: str) -> ComplexityProfile:
    q = query.lower()

    length_score = min(len(query) / 600, 1.0) * 3.0
    requirement_count = sum([
        q.count(" and "), q.count(" also "),
        q.count(" additionally "), q.count(" plus "),
        q.count(" as well as "), q.count(", then "),
    ])
    req_score = min(requirement_count, 3) * 0.7
    deep_tech = sum(1 for term in DEEP_TECH_TERMS if term in q)
    deep_tech_score = min(deep_tech, 3) * 0.8
    abstraction = sum(1 for term in ABSTRACTION_TERMS if term in q)
    abstraction_score = min(abstraction, 2) * 0.8
    expects_long = any(phrase in q for phrase in [
        "full implementation", "complete", "entire",
        "from scratch", "end to end", "whole program",
        "all the", "every", "comprehensive",
    ])
    long_output_score = 1.5 if expects_long else 0.0
    total = length_score + req_score + deep_tech_score + abstraction_score + long_output_score

    matched_hard = any(phrase in q for phrase in HARD_PHRASES)
    matched_easy = any(phrase in q for phrase in EASY_PHRASES) and len(query) < 150

    if matched_hard and matched_easy and total < 2.5:
        # A HARD_PHRASES term (e.g. "neural network") shows up in what's
        # otherwise a short, simple "what is X" / "define X" question with
        # no other complexity signals (no multi-part asks, no expects-long-
        # output, etc). Topic vocabulary alone shouldn't outweigh a
        # genuinely low complexity score — confirmed false positive:
        # "What is a neural network?" was costing real remote tokens on a
        # question local answers just as well (see README section 6/7).
        # Queries that are BOTH easy-phrased AND genuinely complex (e.g.
        # "...then explain time complexity...") keep total >= 2.5 and fall
        # through to the hard-phrase branch below, unaffected.
        rule_decision = "local"
        reason = f"matched easy phrase + short, hard-phrase term outweighed by low complexity ({total:.1f})"
    elif matched_hard:
        rule_decision = "remote"
        reason = "matched hard phrase"
    elif matched_easy:
        rule_decision = "local"
        reason = "matched easy phrase + short"
    elif total >= CONFIG.COMPLEXITY_REMOTE_THRESHOLD:
        rule_decision = "remote"
        reason = f"complexity score {total:.1f} >= threshold"
    elif total < 2.5:
        rule_decision = "local"
        reason = f"complexity score {total:.1f} clearly low"
    else:
        rule_decision = "uncertain"
        reason = f"complexity score {total:.1f} in grey zone"

    return ComplexityProfile(
        query_length=len(query),
        requirement_count=requirement_count,
        deep_tech_signals=deep_tech,
        abstraction_signals=abstraction,
        expects_long_output=expects_long,
        total_score=round(total, 2),
        rule_decision=rule_decision,
        reason=reason,
    )


def _rule_fallback(fallback_score: float, reason: str) -> tuple[str, float, str]:
    midpoint = CONFIG.COMPLEXITY_REMOTE_THRESHOLD / 2
    fallback_decision = "remote" if fallback_score >= midpoint else "local"
    return fallback_decision, 0.0, reason


def _ml_classify(query: str, fallback_score: float) -> tuple[str, float, str]:
    """
    Runs the local zero-shot classifier for grey-zone queries.

    CHANGED FROM ORIGINAL SPEC: if the classifier throws (model failed to
    download, OOM, etc.), we no longer blindly default to "remote" every
    time. Instead we fall back to the rule-based score compared against the
    midpoint of the local/remote zone — still safety-biased, but it won't
    silently burn tokens on every classifier hiccup.

    ALSO CHANGED: the classifier's first call downloads a model from
    HuggingFace. If the network can't reach it (blocked/reset connection),
    that call blocks forever with no exception ever raised — the except
    below wouldn't help. So the load + inference runs on a daemon thread
    with a hard timeout (CONFIG.CLASSIFIER_TIMEOUT); if it doesn't finish
    in time, we give up and use the rule-based fallback instead of hanging
    the whole app on one query.
    """
    result_holder: dict = {}

    def worker():
        try:
            clf = get_classifier()
            result_holder["result"] = clf(
                query,
                candidate_labels=[
                    "simple straightforward coding task",
                    "complex advanced coding task requiring expertise",
                ],
            )
        except Exception as e:
            result_holder["error"] = e

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(timeout=CONFIG.CLASSIFIER_TIMEOUT)

    if thread.is_alive():
        logger.warning(
            f"ML classifier timed out after {CONFIG.CLASSIFIER_TIMEOUT}s "
            f"(likely stuck downloading model / network issue). Falling back to rule score."
        )
        return _rule_fallback(fallback_score, "classifier_timeout_fallback")

    if "error" in result_holder:
        logger.warning(f"ML classifier failed: {result_holder['error']}. Falling back to rule score.")
        return _rule_fallback(fallback_score, "classifier_error_fallback")

    result = result_holder["result"]
    top_label = result["labels"][0]
    top_score = result["scores"][0]
    if "simple" in top_label and top_score >= CONFIG.CONFIDENCE_THRESHOLD:
        return "local", top_score, top_label
    else:
        return "remote", top_score, top_label


def route(query: str) -> ComplexityProfile:
    profile = _compute_complexity(query)

    if profile.rule_decision != "uncertain":
        profile.final_decision = profile.rule_decision
        logger.info(f"ROUTE [{profile.final_decision.upper()}] score={profile.total_score} | {profile.reason}")
        return profile

    ml_decision, ml_conf, ml_label = _ml_classify(query, profile.total_score)
    profile.ml_confidence = ml_conf
    profile.ml_label = ml_label
    profile.final_decision = ml_decision
    profile.reason += f" | ML: {ml_label} ({ml_conf:.0%})"
    logger.info(f"ROUTE [{profile.final_decision.upper()}] score={profile.total_score} | {profile.reason}")
    return profile
