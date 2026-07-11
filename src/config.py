"""
config.py — every tunable knob for RouteWise lives here.

WHY THIS FILE EXISTS:
Instead of hardcoding numbers like "800" or "0.72" deep inside other files,
we put them all here. When you're tuning your score during the hackathon,
this is the ONLY file you should need to touch.
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    # ---- Local Model (runs on AMD MI300X via Ollama, costs ZERO score-tokens) ----
    LOCAL_MODEL: str = "qwen3:8b"
    LOCAL_MAX_TOKENS: int = 512      # how long local's answer is allowed to be
    LOCAL_TEMPERATURE: float = 0.1   # low = more deterministic/focused answers
    LOCAL_TIMEOUT: int = 60          # seconds before we give up on local (observed
                                      # legit answers taking up to ~40s with qwen3:8b)

    # ---- Remote Model (Fireworks AI — every token here IS your score) ----
    FIREWORKS_API_KEY: str = os.getenv("FIREWORKS_API_KEY", "")
    FIREWORKS_BASE_URL: str = "https://api.fireworks.ai/inference/v1/chat/completions"
    REMOTE_MODEL: str = "accounts/fireworks/models/gpt-oss-120b"
    REMOTE_MAX_TOKENS: int = 1500    # hard cap so remote can't ramble and burn tokens.
                                      # Raised from 800 after direct testing showed ALL
                                      # 12 hard-labeled baseline queries hitting
                                      # finish_reason="length" (real truncation, not just
                                      # theoretical) -- one case returned syntactically
                                      # broken code cut off mid-statement. 1500 is a
                                      # deliberate middle ground: it fixes 4/12 (the ones
                                      # that needed <1500 completion tokens) for a bounded
                                      # +51% token cost. The other 8/12 still truncate --
                                      # full coverage would need ~2500+ (2 of them didn't
                                      # even finish at a 2500 probe cap: "design a full
                                      # microservice architecture", "distributed message
                                      # queue with sharding+replication" -- genuinely
                                      # asking for an entire system in one call) and would
                                      # roughly double total tokens again. See README
                                      # section 7 for the accepted tradeoff.
    REMOTE_TEMPERATURE: float = 0.1

    # ---- Router Thresholds (the 3 dials you'll tune during eval) ----
    COMPLEXITY_REMOTE_THRESHOLD: float = 7.5   # raise = fewer remote calls, riskier accuracy
    CONFIDENCE_THRESHOLD: float = 0.72         # raise = escalate more, safer accuracy
    CASCADE_ENABLED: bool = True               # pass local's failed attempt to remote to fix

    # ---- Router safety net (not one of the 3 tuning dials) ----
    # Grey-zone routing loads a HuggingFace model on first use. If the
    # network can't reach HuggingFace, don't hang the whole app waiting on
    # the download — give up after this many seconds and use the rule score.
    CLASSIFIER_TIMEOUT: int = 10

    # ---- Caching ----
    CACHE_ENABLED: bool = True
    CACHE_FILE: str = "cache_store/query_cache.json"

    # ---- Semantic cache matching (optional — requires requirements-ml.txt) ----
    # Catches differently-phrased-but-equivalent queries, not just exact
    # text matches. Gracefully falls back to exact-match-only if
    # sentence-transformers isn't installed (same pattern as router.py's
    # grey-zone classifier). Threshold is deliberately high — a false-
    # positive cache hit serves a WRONG cached answer, which is worse for
    # accuracy than just missing the cache and re-answering.
    CACHE_SEMANTIC_MATCHING: bool = True
    CACHE_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    CACHE_SIMILARITY_THRESHOLD: float = 0.90

    # ---- Logging ----
    LOG_FILE: str = "logs/routewise.log"
    LOG_LEVEL: str = "INFO"

    # ---- Scoring ----
    ACCURACY_THRESHOLD: float = 0.80  # below this, accuracy fails regardless of tokens


CONFIG = Config()
