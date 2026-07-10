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
    LOCAL_TIMEOUT: int = 30          # seconds before we give up on local

    # ---- Remote Model (Fireworks AI — every token here IS your score) ----
    FIREWORKS_API_KEY: str = os.getenv("FIREWORKS_API_KEY", "")
    FIREWORKS_BASE_URL: str = "https://api.fireworks.ai/inference/v1/chat/completions"
    REMOTE_MODEL: str = "accounts/fireworks/models/gpt-oss-120b"
    REMOTE_MAX_TOKENS: int = 800     # hard cap so remote can't ramble and burn tokens
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

    # ---- Logging ----
    LOG_FILE: str = "logs/routewise.log"
    LOG_LEVEL: str = "INFO"

    # ---- Scoring ----
    ACCURACY_THRESHOLD: float = 0.80  # below this, accuracy fails regardless of tokens


CONFIG = Config()
