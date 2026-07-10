"""
cache.py — Layer 1: identical (and now, near-identical) queries return an
instant, free answer.

WHY: Hackathon evaluators often run test suites with repeated or near-repeated
prompts. Without a cache, you'd pay remote tokens every single time. With it,
you pay once, ever, per unique-enough query — and the cache survives restarts
because it's saved to disk as JSON.

TWO MATCHING STAGES:
  1. Exact match (hash of normalized text) — instant, always available, no
     model needed.
  2. Semantic match (embedding cosine similarity) — catches differently-
     phrased but equivalent queries, e.g. "sum two numbers" vs "add two
     numbers". Requires sentence-transformers (requirements-ml.txt);
     gracefully falls back to exact-match-only if that's not installed or
     the model can't be reached, same pattern as router.py's grey-zone
     classifier. Threshold is deliberately high (CONFIG.CACHE_SIMILARITY_
     THRESHOLD) — a false-positive semantic hit serves a WRONG cached
     answer, which is worse for accuracy than just missing the cache.
"""

import hashlib
import json
import os
import threading
from typing import Optional
from src.config import CONFIG
from src.logger import get_logger

logger = get_logger("cache")

_embedder = None
_embedder_unavailable = False  # set True after one failed/timed-out attempt so
                                # we don't re-try loading on every single query


def _get_embedder():
    """Lazy-loads the sentence-transformers model once, reuses it after that."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading sentence-transformers model '{CONFIG.CACHE_EMBEDDING_MODEL}' for semantic cache matching...")
        _embedder = SentenceTransformer(CONFIG.CACHE_EMBEDDING_MODEL)
        logger.info("✅ Semantic cache matching enabled")
    return _embedder


def _embed_with_timeout(text: str):
    """
    Runs embedding on a daemon thread with a hard timeout — same pattern as
    router.py's classifier. The model downloads on first use; on a network
    that blocks/resets the connection (we hit exactly this earlier in this
    project with the grey-zone classifier), a bare call would hang forever.
    On timeout OR failure, semantic matching is disabled for the rest of the
    process — this runs on every cache lookup, not just grey-zone queries,
    so silently eating a 10s timeout on every single query would be a much
    bigger cost than the one-time classifier case.
    """
    global _embedder_unavailable
    if _embedder_unavailable:
        return None

    result_holder: dict = {}

    def worker():
        try:
            embedder = _get_embedder()
            result_holder["vector"] = embedder.encode(text, normalize_embeddings=True)
        except Exception as e:
            result_holder["error"] = e

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(timeout=CONFIG.CLASSIFIER_TIMEOUT)

    if thread.is_alive():
        logger.warning(
            f"Semantic cache embedding timed out after {CONFIG.CLASSIFIER_TIMEOUT}s "
            f"(likely network issue downloading model). Disabling semantic cache "
            f"matching for this session — falling back to exact-match only."
        )
        _embedder_unavailable = True
        return None

    if "error" in result_holder:
        logger.warning(f"Semantic cache embedding failed: {result_holder['error']}. Falling back to exact-match only.")
        _embedder_unavailable = True
        return None

    return result_holder.get("vector")


def _cosine_similarity(a, b) -> float:
    import numpy as np
    # Both vectors are already unit-normalized (normalize_embeddings=True),
    # so the dot product IS the cosine similarity — no extra division needed.
    return float(np.dot(np.array(a), np.array(b)))


class QueryCache:
    def __init__(self):
        os.makedirs(os.path.dirname(CONFIG.CACHE_FILE), exist_ok=True)
        self.cache: dict = {}
        self.hits: int = 0
        self.semantic_hits: int = 0
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

        # Stage 1: exact match — free, instant, no model needed.
        key = self._key(query)
        result = self.cache.get(key)
        if result:
            self.hits += 1
            logger.info(f"✅ CACHE HIT (exact, total exact hits: {self.hits})")
            return {**result, "match_type": "exact"}

        # Stage 2: semantic match — only queries that missed exact match,
        # only if enabled, and only if there's anything to compare against.
        if CONFIG.CACHE_SEMANTIC_MATCHING and self.cache:
            query_vector = _embed_with_timeout(query)
            if query_vector is not None:
                best_score, best_entry = 0.0, None
                for entry in self.cache.values():
                    if "embedding" not in entry:
                        continue
                    score = _cosine_similarity(query_vector, entry["embedding"])
                    if score > best_score:
                        best_score, best_entry = score, entry
                if best_entry and best_score >= CONFIG.CACHE_SIMILARITY_THRESHOLD:
                    self.semantic_hits += 1
                    logger.info(
                        f"✅ CACHE HIT (semantic, {best_score:.0%} similar to "
                        f'"{best_entry["query_preview"]}", total semantic hits: {self.semantic_hits})'
                    )
                    return {**best_entry, "match_type": "semantic", "match_similarity": best_score}

        self.misses += 1
        return None

    def set(self, query: str, answer: str, source: str, tokens_used: int = 0):
        key = self._key(query)
        entry = {
            "answer": answer,
            "source": source,
            "tokens_used": tokens_used,
            "query_preview": query[:80],
        }
        if CONFIG.CACHE_SEMANTIC_MATCHING:
            vector = _embed_with_timeout(query)
            if vector is not None:
                entry["embedding"] = vector.tolist()
        self.cache[key] = entry
        self._save()

    def stats(self) -> dict:
        total_hits = self.hits + self.semantic_hits
        return {
            "total_entries": len(self.cache),
            "hits": self.hits,
            "semantic_hits": self.semantic_hits,
            "misses": self.misses,
            "hit_rate": f"{total_hits / max(total_hits + self.misses, 1) * 100:.1f}%",
        }

    def clear(self):
        self.cache = {}
        self._save()


cache = QueryCache()
