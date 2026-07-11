"""
verifier.py — Layer 4: is local's answer actually trustworthy?

This is a pure heuristic checker — NO model call here, because calling a
model to verify a model's answer would defeat the whole point of being
token-efficient. Everything below is string/AST inspection, which is free.
"""

import ast
import re
from dataclasses import dataclass
from typing import Optional
from src.logger import get_logger

logger = get_logger("verifier")


@dataclass
class VerificationResult:
    is_confident: bool
    failures: list[str]
    confidence_score: float
    verdict: str


# Unambiguous imperatives — "write/implement/debug X" is always a code ask,
# regardless of how the rest of the sentence is phrased.
_IMPERATIVE_VERBS = ["write", "implement", "create", "generate", "debug", "refactor", "fix"]

# These words show up just as often in conceptual questions ("how does a
# hash function work", "explain how a build pipeline works") as they do in
# real code requests, so they only count when the query ISN'T phrased as
# an explanation.
_AMBIGUOUS_CODE_WORDS = ["function", "class", "program", "script", "algorithm", "build", "code"]

_EXPLAIN_MARKERS = [
    "what is", "what's", "what are", "what does", "what do",
    "how does", "how is", "how are", "why does", "why is",
    "explain", "describe", "define", "difference between",
]

_IMPERATIVE_RE = re.compile(r"\b(" + "|".join(_IMPERATIVE_VERBS) + r")\b", re.IGNORECASE)
_AMBIGUOUS_RE = re.compile(r"\b(" + "|".join(_AMBIGUOUS_CODE_WORDS) + r")\b", re.IGNORECASE)


def wants_code(query: str) -> bool:
    """True if the query is actually asking to produce code, not just an
    explanation that happens to mention a code-ish word."""
    q_lower = query.lower()
    if _IMPERATIVE_RE.search(q_lower):
        return True
    if _AMBIGUOUS_RE.search(q_lower) and not any(m in q_lower for m in _EXPLAIN_MARKERS):
        return True
    return False


REFUSAL_PHRASES = [
    "i cannot", "i can't", "i'm not able", "i am not able",
    "i don't know", "i'm unable", "beyond my capabilities",
    "i'm not sure", "i don't have enough", "i apologize",
    "as an ai", "i lack the", "this is too complex",
]

PLACEHOLDER_MARKERS = [
    "# TODO", "# todo", "# YOUR CODE HERE", "# your code here",
    "...", "pass  #", "raise NotImplementedError",
    "[INSERT", "[ADD", "/* TODO */", "// TODO",
]

_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def _extract_python_block(answer: str) -> Optional[str]:
    """Pulls the first fenced code block out of the answer, if any."""
    match = _CODE_FENCE_RE.search(answer)
    if match:
        return match.group(1)
    return None


def _has_real_syntax_error(answer: str, query_lower: str) -> bool:
    """
    CHANGED FROM ORIGINAL SPEC: instead of guessing from colon counts,
    we try to actually PARSE the code with Python's ast module. If there's
    a fenced code block, parse that; otherwise, if the whole answer looks
    like bare Python (has 'def '/'class ' at line start), try parsing the
    whole thing. Non-Python answers are skipped entirely.
    """
    if "python" not in query_lower and "def " not in answer:
        return False  # not a python-code situation, nothing to check

    code = _extract_python_block(answer) or answer
    if "def " not in code and "class " not in code:
        return False  # no function/class definitions to worry about

    try:
        ast.parse(code)
        return False
    except SyntaxError:
        return True
    except Exception:
        # Anything unexpected (e.g. non-code text) — don't penalize for it.
        return False


def verify(answer: Optional[str], query: str) -> VerificationResult:
    if not answer or len(answer.strip()) == 0:
        return VerificationResult(
            is_confident=False,
            failures=["empty_answer"],
            confidence_score=0.0,
            verdict="❌ Empty answer — escalating",
        )

    failures = []
    q_lower = query.lower()
    a_lower = answer.lower()

    has_code = (
        "def " in answer or "class " in answer or
        "```" in answer or "function " in answer or
        "return " in answer
    )

    # A short answer that's already real code (e.g. a one-line print/lambda
    # solution) isn't a red flag — only penalize shortness when there's no
    # code content backing it up (likely a stub or non-answer).
    if len(answer.strip()) < 60 and not has_code:
        failures.append("too_short")
    if any(phrase in a_lower for phrase in REFUSAL_PHRASES):
        failures.append("model_refused")
    if any(marker in answer for marker in PLACEHOLDER_MARKERS):
        failures.append("has_placeholders")

    if wants_code(query) and not has_code:
        failures.append("missing_code")

    if (len(answer) < len(query) * 1.3 and len(query) > 50
            and answer[:80].lower() in q_lower):
        failures.append("restating_question")

    if _has_real_syntax_error(answer, q_lower):
        failures.append("broken_python_syntax")

    stripped = answer.strip()
    if len(stripped) > 100 and not any(stripped.endswith(end) for end in [
        ".", "```", "}", "]", ")", "\n", ":", ";", '"', "'"
    ]):
        failures.append("truncated_output")

    penalty_map = {
        "empty_answer": 1.0, "model_refused": 0.9,
        "has_placeholders": 0.5, "missing_code": 0.6,
        "restating_question": 0.4, "broken_python_syntax": 0.7,
        "truncated_output": 0.3, "too_short": 0.2,
    }
    total_penalty = sum(penalty_map.get(f, 0.2) for f in failures)
    confidence_score = max(0.0, 1.0 - total_penalty)
    is_confident = len(failures) == 0

    verdict = (
        f"✅ LOCAL answer accepted (confidence: {confidence_score:.0%})"
        if is_confident else
        f"❌ Escalating. Failures: {', '.join(failures)}"
    )
    logger.info(verdict)
    return VerificationResult(
        is_confident=is_confident,
        failures=failures,
        confidence_score=confidence_score,
        verdict=verdict,
    )
