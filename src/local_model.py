"""
local_model.py — Layer 3: always try the free local model first
(unless the router already skipped straight to remote).

Runs via Ollama, talking to codellama:7b on your AMD MI300X box.
A failed local attempt costs ZERO tokens — that's the whole point of
trying it first even when we're not fully sure it'll succeed.
"""

import time
from typing import Optional
import ollama
from src.config import CONFIG
from src.logger import get_logger

logger = get_logger("local_model")

# CONFIG.LOCAL_TIMEOUT was previously unused — ollama.chat() had no timeout,
# so an unresponsive Ollama server would hang the whole app forever. A
# dedicated Client with a timeout catches that case; a normal slow-but-working
# answer still completes fine as long as it's under LOCAL_TIMEOUT.
_client = ollama.Client(timeout=CONFIG.LOCAL_TIMEOUT)


def build_prompt(query: str) -> str:
    return f"""You are an expert programmer. Be concise and precise.

Task: {query}

Requirements:
- Provide working, complete code
- Include only necessary comments
- No lengthy explanations unless specifically asked
- If it's a concept question, answer in 2-3 sentences max

Solution:"""


def call_local(query: str) -> tuple[Optional[str], float]:
    prompt = build_prompt(query)
    start = time.time()
    try:
        response = _client.chat(
            model=CONFIG.LOCAL_MODEL,
            messages=[{"role": "user", "content": prompt}],
            think=False,  # qwen3 is a reasoning model; without this, its answer
                          # goes into a separate "thinking" field and content is empty
            options={
                "temperature": CONFIG.LOCAL_TEMPERATURE,
                "num_predict": CONFIG.LOCAL_MAX_TOKENS,
            },
        )
        latency = time.time() - start
        answer = response["message"]["content"].strip()
        logger.info(f"Local response: {len(answer)} chars in {latency:.1f}s")
        return answer, latency
    except Exception as e:
        latency = time.time() - start
        logger.warning(f"Local model error after {latency:.1f}s: {e}")
        return None, latency


def is_local_available() -> bool:
    try:
        models = _client.list()
        available = [m.model for m in models.models]
        if CONFIG.LOCAL_MODEL in available or any(CONFIG.LOCAL_MODEL in m for m in available):
            logger.info(f"✅ Local model '{CONFIG.LOCAL_MODEL}' is available")
            return True
        else:
            logger.warning(f"⚠️ Model '{CONFIG.LOCAL_MODEL}' not found. Run: ollama pull {CONFIG.LOCAL_MODEL}")
            return False
    except Exception as e:
        logger.error(f"❌ Ollama not reachable: {e}")
        return False
