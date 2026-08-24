"""HTTP client for the Modal retrieval-as-a-service (CPU-only).

Mirrors the local tool callables (keyword_search / semantic_search /
chunk_read / corpus_map) but dispatches them to the deployed Modal web
endpoint instead of running the embedder locally. This keeps BGE-M3 +
bge-reranker off the generation box (which has only ~3.2 GB RAM).
"""

from __future__ import annotations

import requests


class RemoteRetrieval:
    def __init__(self, base_url: str, timeout: float = 180.0):
        self.base = base_url.rstrip("/")
        self.timeout = timeout

    def _post(self, name: str, payload):
        r = requests.post(f"{self.base}/{name}", json=payload or {}, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

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

    def resolve_sources(self, refs):
        return self._post("resolve_sources", {"refs": list(refs or [])})
