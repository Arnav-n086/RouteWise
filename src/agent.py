"""
agent.py — orchestrates all 5 layers for a single query.

FLOW:
  Layer 1: cache check           -> instant return if seen before (0 tokens)
  Layer 2: complexity routing    -> decide local vs remote vs "try local first"
  Layer 3: try local             -> if router didn't skip straight to remote
  Layer 4: verify local's answer -> confident? return it (0 tokens)
  Layer 5: targeted remote call  -> only if local failed verification
"""

import time
from dataclasses import dataclass, field
from typing import Optional

from src.config import CONFIG
from src.cache import cache
from src.router import route, ComplexityProfile
from src.local_model import call_local
from src.remote_model import call_remote, get_token_stats
from src.verifier import verify, VerificationResult
from src.token_tracker import tracker
from src.logger import get_logger

logger = get_logger("agent")


@dataclass
class AgentResult:
    query: str
    answer: str
    remote_tokens_used: int = 0
    served_from: str = ""
    path_taken: list[str] = field(default_factory=list)
    complexity_profile: Optional[ComplexityProfile] = None
    verification: Optional[VerificationResult] = None
    total_latency: float = 0.0
    local_latency: float = 0.0
    remote_latency: float = 0.0
    success: bool = True
    error: Optional[str] = None

    def summary(self) -> str:
        return (
            f"[{self.served_from.upper()}] "
            f"tokens={self.remote_tokens_used} | "
            f"latency={self.total_latency:.1f}s | "
            f"path: {' → '.join(self.path_taken)}"
        )


def run(query: str) -> AgentResult:
    start_time = time.time()
    result = AgentResult(query=query, answer="")
    logger.info(f"\n{'='*60}")
    logger.info(f"QUERY: {query[:80]}{'...' if len(query) > 80 else ''}")

    # ---------------- Layer 1: Cache ----------------
    cached = cache.get(query)
    if cached:
        result.answer = cached["answer"]
        result.served_from = "cache"
        result.remote_tokens_used = 0
        if cached.get("match_type") == "semantic":
            similarity = cached["match_similarity"]
            note = f'semantic match ({similarity:.0%}) to: "{cached["query_preview"]}"'
            result.path_taken = [f"CACHE HIT (semantic, {similarity:.0%} similar)"]
        else:
            note = "exact match"
            result.path_taken = ["CACHE HIT (exact)"]
        tracker.record(query, "cache", 0, note=note)
        result.total_latency = time.time() - start_time
        return result

    result.path_taken.append("cache_miss")

    # ---------------- Layer 2: Complexity routing ----------------
    profile = route(query)
    result.complexity_profile = profile
    result.path_taken.append(f"complexity={profile.total_score:.1f}")

    if profile.final_decision == "remote" and profile.total_score >= CONFIG.COMPLEXITY_REMOTE_THRESHOLD:
        # Router is confident this is hard enough to skip local entirely.
        result.path_taken.append("SKIP_LOCAL→REMOTE")
        answer, tokens, latency = call_remote(query)
        result.remote_latency = latency
        result.remote_tokens_used = tokens
        if answer:
            result.answer = answer
            result.served_from = "remote"
            cache.set(query, answer, "remote", tokens)
        else:
            result.success = False
            result.error = "Remote call failed"
            result.answer = "Error: Could not generate answer"
        result.total_latency = time.time() - start_time
        return result

    # ---------------- Layer 3: Try local ----------------
    result.path_taken.append("TRY_LOCAL")
    local_answer, local_latency = call_local(query)
    result.local_latency = local_latency

    if local_answer is None:
        result.path_taken.append("LOCAL_CRASH→REMOTE")
        answer, tokens, latency = call_remote(query)
        result.remote_latency = latency
        result.remote_tokens_used = tokens
        result.answer = answer or "Error: All models failed"
        result.served_from = "remote"
        result.success = bool(answer)
        if answer:
            cache.set(query, answer, "remote", tokens)
        result.total_latency = time.time() - start_time
        return result

    tracker.record(query, "local", 0, note=f"local attempt, {len(local_answer)} chars")

    # ---------------- Layer 4: Verify local ----------------
    result.path_taken.append("VERIFY_LOCAL")
    verification = verify(local_answer, query)
    result.verification = verification

    if verification.is_confident:
        result.answer = local_answer
        result.served_from = "local"
        result.remote_tokens_used = 0
        result.path_taken.append("LOCAL_ACCEPTED✅")
        cache.set(query, local_answer, "local", 0)
        result.total_latency = time.time() - start_time
        return result

    # ---------------- Layer 5: Targeted remote fallback ----------------
    local_attempt_to_pass = local_answer if CONFIG.CASCADE_ENABLED else None
    result.path_taken.append(f"ESCALATE→REMOTE({','.join(verification.failures)})")
    answer, tokens, latency = call_remote(query, local_attempt=local_attempt_to_pass)
    result.remote_latency = latency
    result.remote_tokens_used = tokens
    result.served_from = "local→remote"

    if answer:
        result.answer = answer
        result.success = True
        cache.set(query, answer, "remote", tokens)
    else:
        result.answer = local_answer
        result.served_from = "local_fallback"
        result.success = False
        result.error = "Remote failed, using local fallback"

    result.total_latency = time.time() - start_time
    return result


def session_stats() -> dict:
    return {**get_token_stats(), **cache.stats(), **tracker.summary()}
