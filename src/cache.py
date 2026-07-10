"""
cache.py — Layer 1: identical queries return an instant, free answer.

WHY: Hackathon evaluators often run test suites with repeated or near-repeated
prompts. Without a cache, you'd pay remote tokens every single time. With it,
you pay once, ever, per unique query — and the cache survives restarts because
it's saved to disk as JSON.
"""

import hashlib
import json
import os
from typing import Optional
from src.config import CONFIG
from src.logger import get_logger

logger = get_logger("cache")


class QueryCache:
    def __init__(self):
        os.makedirs(os.path.dirname(CONFIG.CACHE_FILE), exist_ok=True)
        self.cache: dict = {}
        self.hits: int = 0
        self.misses: int = 0
        self._load()

    def _load(self):
        if os.path.exists(CONFIG.CACHE_FILE):
            try:
                with open(CONFIG.CACHE_FILE, "r") as f:
                    self.cache = json.load(f)
                logger.info(f"Cache loaded: {len(self.cache)} entries")
            except Exception as e:
                logger.warning(f"Cache load failed: {e}. Starting fresh.")
                self.cache = {}

    def _save(self):
        try:
            with open(CONFIG.CACHE_FILE, "w") as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            logger.error(f"Cache save failed: {e}")

    def _key(self, query: str) -> str:
        # Normalize (strip whitespace, lowercase) so "Write a func" and
        # "write a func " hash to the SAME cache entry.
        normalized = query.strip().lower()
        return hashlib.md5(normalized.encode()).hexdigest()

    def get(self, query: str) -> Optional[dict]:
        if not CONFIG.CACHE_ENABLED:
            return None
        key = self._key(query)
        result = self.cache.get(key)
        if result:
            self.hits += 1
            logger.info(f"✅ CACHE HIT (total hits: {self.hits})")
            return result
        self.misses += 1
        return None

    def set(self, query: str, answer: str, source: str, tokens_used: int = 0):
        key = self._key(query)
        self.cache[key] = {
            "answer": answer,
            "source": source,
            "tokens_used": tokens_used,
            "query_preview": query[:80],
        }
        self._save()

    def stats(self) -> dict:
        return {
            "total_entries": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{self.hits / max(self.hits + self.misses, 1) * 100:.1f}%",
        }

    def clear(self):
        self.cache = {}
        self._save()


cache = QueryCache()
