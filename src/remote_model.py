"""
remote_model.py — Layer 5: the ONLY place actual score-tokens get spent.

Every call here talks to Fireworks AI over FIREWORKS_BASE_URL. We track
tokens two ways:
  1. A simple running total (kept for backwards compatibility / quick checks)
  2. A full event log via token_tracker (added so you can see WHERE tokens
     went, not just the final number)
"""

import time
import requests
from typing import Optional
from src.config import CONFIG
from src.logger import get_logger
from src.token_tracker import tracker
from src.verifier import wants_code

logger = get_logger("remote_model")

_total_remote_tokens = 0
_total_remote_calls = 0


def get_token_stats() -> dict:
    return {
        "total_remote_tokens": _total_remote_tokens,
        "total_remote_calls": _total_remote_calls,
        "avg_tokens_per_call": round(_total_remote_tokens / max(_total_remote_calls, 1), 1),
    }


def build_remote_prompt(query: str, local_attempt: Optional[str] = None) -> str:
    code_mode = wants_code(query)
    if local_attempt and len(local_attempt.strip()) > 60:
        # CASCADE: ask remote to FIX local's attempt rather than redo it from
        # scratch. A "fix this" prompt + a shorter expected answer = fewer
        # tokens than "solve this from zero" would cost.
        trimmed_attempt = local_attempt[:600]
        if code_mode:
            return f"""Fix the following code solution. Return only corrected, working code. No explanations.

Task: {query}

Broken/incomplete attempt:
{trimmed_attempt}

Corrected solution:"""
        return f"""The previous answer to this question was incomplete or unclear. Give a corrected, direct answer in plain text — no code unless the question asks for it.

Question: {query}

Previous attempt:
{trimmed_attempt}

Corrected answer:"""
    else:
        if code_mode:
            return f"""Solve this coding task. Return only working code. No explanations unless asked.

Task: {query}

Solution:"""
        return f"""Answer this question directly and clearly in plain text. Do not include code unless the question explicitly asks for it.

Question: {query}

Answer:"""


def call_remote(query: str, local_attempt: Optional[str] = None) -> tuple[Optional[str], int, float]:
    global _total_remote_tokens, _total_remote_calls

    if not CONFIG.FIREWORKS_API_KEY:
        logger.error("FIREWORKS_API_KEY not set.")
        return None, 0, 0.0

    prompt = build_remote_prompt(query, local_attempt)
    start = time.time()
    try:
        response = requests.post(
            CONFIG.FIREWORKS_BASE_URL,
            headers={
                "Authorization": f"Bearer {CONFIG.FIREWORKS_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": CONFIG.REMOTE_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": CONFIG.REMOTE_MAX_TOKENS,
                "temperature": CONFIG.REMOTE_TEMPERATURE,
            },
            timeout=60,
        )
        latency = time.time() - start
        if response.status_code != 200:
            logger.error(f"Fireworks API error {response.status_code}: {response.text}")
            tracker.record(query, "remote", 0, note=f"API error {response.status_code}")
            return None, 0, latency

        data = response.json()
        answer = data["choices"][0]["message"]["content"].strip()
        usage = data.get("usage", {})
        tokens_used = usage.get("total_tokens", 0)

        _total_remote_tokens += tokens_used
        _total_remote_calls += 1
        mode = "FIX (cascade)" if local_attempt else "FRESH"
        tracker.record(query, "remote", tokens_used, note=f"{mode} | model={CONFIG.REMOTE_MODEL}")

        logger.info(f"💸 Remote: {tokens_used} tokens | SESSION TOTAL: {_total_remote_tokens}")
        return answer, tokens_used, latency

    except requests.Timeout:
        latency = time.time() - start
        logger.error(f"Remote call timed out after {latency:.1f}s")
        tracker.record(query, "remote", 0, note="TIMEOUT")
        return None, 0, latency
    except Exception as e:
        latency = time.time() - start
        logger.error(f"Remote call failed: {e}")
        tracker.record(query, "remote", 0, note=f"ERROR: {e}")
        return None, 0, latency
