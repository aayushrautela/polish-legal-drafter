"""HTTP client for the retrieval tool service.

Mirrors the local tool callables (keyword_search / semantic_search /
chunk_read / get_template) but dispatches them to a served retrieval
endpoint instead of running the embedder locally. This keeps BGE-M3 +
bge-reranker off the generation box (which has only ~3.2 GB RAM).
Point at a running server (see scripts/rag/server.py), e.g.
http://127.0.0.1:10100.
"""

from __future__ import annotations

import logging
import time
import requests

log = logging.getLogger(__name__)


class RemoteRetrieval:
    def __init__(self, base_url: str, timeout: float = 180.0, max_retries: int = 3):
        self.base = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries

    def _post(self, name: str, payload):
        last_exc = None
        for attempt in range(self.max_retries):
            try:
                r = requests.post(f"{self.base}/{name}", json=payload or {}, timeout=self.timeout)
                r.raise_for_status()
                return r.json()
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_exc = exc
                wait = min(2 ** attempt, 10)
                log.warning("[remote] %s attempt %d failed: %s (retrying in %ds)", name, attempt + 1, exc, wait)
                time.sleep(wait)
            except requests.HTTPError as exc:
                last_exc = exc
                if exc.response is not None and exc.response.status_code >= 500:
                    wait = min(2 ** attempt, 10)
                    log.warning("[remote] %s attempt %d HTTP %d (retrying in %ds)", name, attempt + 1, exc.response.status_code, wait)
                    time.sleep(wait)
                else:
                    raise
        raise last_exc

    def keyword_search(self, **kw):
        return self._post("keyword_search", kw)

    def semantic_search(self, **kw):
        return self._post("semantic_search", kw)

    def chunk_read(self, **kw):
        return self._post("chunk_read", kw)

    def get_template(self, doc_type: str, query: str = None):
        payload = {"doc_type": doc_type}
        if query:
            payload["query"] = query
        return self._post("get_template", payload)

    def health(self) -> dict:
        """GET /health and return the JSON body, retrying transient failures."""
        last_exc = None
        for attempt in range(self.max_retries):
            try:
                r = requests.get(f"{self.base}/health", timeout=self.timeout)
                r.raise_for_status()
                return r.json()
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_exc = exc
                wait = min(2 ** attempt, 10)
                log.warning("[remote] health attempt %d failed: %s (retrying in %ds)",
                            attempt + 1, exc, wait)
                time.sleep(wait)
            except requests.HTTPError as exc:
                last_exc = exc
                if exc.response is not None and exc.response.status_code >= 500:
                    wait = min(2 ** attempt, 10)
                    log.warning("[remote] health attempt %d HTTP %d (retrying in %ds)",
                                attempt + 1, exc.response.status_code, wait)
                    time.sleep(wait)
                else:
                    raise
        raise last_exc

    def check_index_ready(self, min_points: int = 1) -> int:
        """Verify the served index is non-empty; return its point count.

        Raises ``RuntimeError`` if ``/health`` is unreachable or reports an
        empty (below ``min_points``) index, so callers abort instead of
        silently running a whole job with zero retrieval grounding.
        """
        h = self.health()
        raw = h.get("points") if isinstance(h, dict) else None
        if raw is None:
            raise RuntimeError(f"RAG /health missing 'points' field: {h!r}")
        try:
            points = int(raw)
        except (TypeError, ValueError):
            raise RuntimeError(f"RAG /health 'points' is not an int: {raw!r}")
        if points < min_points:
            raise RuntimeError(
                f"RAG index is empty (points={points}); refusing to run without "
                "retrieval grounding")
        return points
