"""Retrieval-as-a-service on Modal (CPU-only, free-tier friendly).

Serves the three A-RAG hierarchical tools (keyword_search / semantic_search /
chunk_read) plus a corpus_map endpoint over HTTP, backed by the existing
BGE-M3 + bge-reranker hybrid index stored on the ``legal-drafter-rag`` Volume.

This keeps the embedding models OFF the generation box (which has only ~3.2 GB
RAM and OOMs when it tries to load BGE-M3 + bge-reranker locally). The local
scenario generator calls retrieval over the network instead.

Design notes:
* CPU-only (no GPU) -> cheap, within the $30/mo free credit for burst usage.
* The Qdrant collection is copied to ephemeral ``/tmp`` on cold start because
  embedded RocksDB needs real local file locking that a Modal distributed
  Volume does not provide reliably (same reason modal_build_index builds on
  /tmp then copies the finished collection onto the Volume).
* Models load from the Volume's ``hf_cache`` (populated by modal_build_index).
* A single ASGI (FastAPI) web app exposes every route under ONE URL, so the
  local client just POSTs to ``<base>/<tool_name>``.

Deploy (persistent endpoint; free credit covers burst usage, $0 idle):
    modal deploy modal_retrieval.py

Then point the generator at it:
    RETRIEVAL_ENDPOINT=https://<app>.modal.run \
      PYTHONPATH=src .venv/bin/python -m legal_drafter.scenario_gen \
      --n 3 --retrieval-url https://<app>.modal.run ...
"""

from __future__ import annotations

import shutil
import sys
import logging
from pathlib import Path

import modal

VOLUME_NAME = "legal-drafter-rag"
VOLUME_MOUNT = "/data"
QDRANT_VOLUME = f"{VOLUME_MOUNT}/qdrant"
QDRANT_LOCAL = "/tmp/qdrant"
HF_CACHE = f"{VOLUME_MOUNT}/hf_cache"
COLLECTION = "legal_corpus"
COLLECTION_TEMPLATES = "templates"

# Fields returned when resolving a source ref to its verbatim corpus chunk.
RESOLVE_KEEP = (
    "chunk_id",
    "source_ref",
    "title",
    "text",
    "source_type",
    "top_category",
    "risk_or_safe",
    "display_address",
    "legal_area",
)

image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(
        "torch",
        "flagembedding==1.4.0",
        "sentence-transformers==6.0.0",
        "transformers",
        "qdrant-client==1.19.0",
        "numpy",
        "huggingface_hub",
        "tqdm",
        "fastapi",
    )
    .add_local_dir("src/legal_drafter", remote_path="/pkg/legal_drafter")
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

app = modal.App(name="legal-drafter-retrieval")


class _RetrievalMixin:
    """Shared body for the CPU and GPU retrieval services."""

    @modal.enter()
    def enter(self):
        sys.path.insert(0, "/pkg")
        # RocksDB-friendly copy of the collection to ephemeral disk.
        src = Path(QDRANT_VOLUME)
        dst = Path(QDRANT_LOCAL)
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

        from legal_drafter.retrieval.embeddings import _bge_model, _reranker
        from legal_drafter.retrieval.hybrid import QdrantHybridStore

        self.store = QdrantHybridStore(collection_name=COLLECTION, path=QDRANT_LOCAL)
        # Open the storage folder ONCE (embedded Qdrant forbids two clients on
        # the same folder); both collections live in /tmp/qdrant, so share it.
        _ = self.store.client
        # Optional real templates collection (built from the CC-BY-4.0
        # legal-templates-multilingual dataset). Only attached if present.
        self.template_store = None
        try:
            tstore = QdrantHybridStore(
                collection_name=COLLECTION_TEMPLATES,
                path=QDRANT_LOCAL,
                client=self.store.client,
            )
            if tstore.exists():
                self.template_store = tstore
        except Exception as exc:
            logging.warning("templates store load failed: %s", exc)
            self.template_store = None
        # Pre-load models once so the first tool call is not extra-slow.
        _bge_model()
        _reranker()

        # Build an in-memory lookup of every legal_corpus chunk (cached in the
        # container) so source resolution can be SERVED FROM MODAL without
        # returning the whole ~17k map over HTTP (which exceeds the response
        # size limit and 500s). The generator sends only the refs it cited.
        self.corpus_map: dict[str, dict] = {}
        nxt = None
        while True:
            pts, nxt = self.store.client.scroll(
                collection_name=self.store.collection_name,
                with_payload=True,
                limit=2000,
                offset=nxt,
            )
            for p in pts:
                pl = p.payload or {}
                if not pl:
                    continue
                for kk in (pl.get("chunk_id"), pl.get("source_ref")):
                    if kk:
                        self.corpus_map[str(kk)] = pl
            if nxt is None:
                break

    @modal.asgi_app()
    def app(self):
        # Method name 'app' keeps the historical CPU URL stable:
        # https://<workspace>--legal-drafter-retrieval-retrievalservice-app.modal.run
        # The GPU class reuses this name under its own class-specific URL.
        from fastapi import FastAPI

        from legal_drafter.retrieval.agentic_tools import (
            chunk_read,
            keyword_search,
            semantic_search,
        )

        import threading
        import inspect

        # Blocking scans (keyword_search scrolls the whole corpus) must not run
        # on the event loop nor concurrently: sync routes + global lock.
        _tool_lock = threading.Lock()

        # LLM-issued arguments arrive sloppy: wrong key names ("query" vs
        # "keywords"), scalar where list expected, string numbers, stray keys.
        # Coerce defensively so a malformed call degrades to empty results
        # instead of HTTP 500 (which previously poisoned whole edit runs).
        def _coerce(fn, item: dict) -> dict:
            sig = inspect.signature(fn)
            params = {p for p in sig.parameters}
            item = dict(item or {})
            if fn is keyword_search:
                kw = item.get("keywords")
                if kw is None and "query" in item:
                    kw, item["query"] = item["query"], None
                if isinstance(kw, str):
                    kw = [kw]
                item["keywords"] = [str(k) for k in (kw or [])][:8]
                if "top_k" in item:
                    try:
                        item["top_k"] = max(1, min(50, int(item["top_k"])))
                    except Exception:
                        item["top_k"] = 10
                if "filters" in item and not isinstance(item["filters"], dict):
                    item["filters"] = None
            elif fn is chunk_read and isinstance(item.get("chunk_ids"), str):
                item["chunk_ids"] = [item["chunk_ids"]]
            if fn is semantic_search and not item.get("query") and item.get("keywords"):
                item["query"] = " ".join(str(k) for k in item["keywords"])
            return {k: v for k, v in item.items() if k in params}

        fa = FastAPI()

        import time as _time
        def _safe(fn, item):
            """Run a tool; on bad/missing args return a machine-readable hint
            (HTTP 200) instead of a 500 - the calling LLM can then retry."""
            _t0 = _time.time()
            print(f"[tool] {fn.__name__} keys={sorted((item or {}).keys())}",
                  flush=True)
            try:
                return fn(self.store, **_coerce(fn, item))
            except TypeError as exc:
                req = [p.name for p in inspect.signature(fn).parameters.values()
                       if p.default is inspect.Parameter.empty and p.name != "self"]
                return [{"error": f"{exc}",
                         "hint": f"{fn.__name__} needs args {req}; you sent {sorted((item or {}).keys())}"}]
            except Exception as exc:
                print(f"[tool] {fn.__name__} EXC {type(exc).__name__}: {exc}",
                      flush=True)
                return [{"error": f"{type(exc).__name__}: {exc}",
                         "hint": "retry once; if it repeats, switch tool"}]
            finally:
                print(f"[tool] {fn.__name__} done {(_time.time()-_t0)*1000:.0f}ms",
                      flush=True)

        @fa.post("/keyword_search")
        def kw(item: dict):
            with _tool_lock:
                return _safe(keyword_search, item)

        @fa.post("/semantic_search")
        def sem(item: dict):
            with _tool_lock:
                return _safe(semantic_search, item)

        @fa.post("/chunk_read")
        def cr(item: dict):
            with _tool_lock:
                return _safe(chunk_read, item)

        @fa.post("/get_template")
        async def gt(item: dict):
            from legal_drafter.retrieval.templates import (
                get_template_real,
                get_template_text,
                template_result,
            )

            dt = (item or {}).get("doc_type", "other")
            text = None
            # These two doc_types resolve to a WRONG sales-agreement entry in
            # the templates collection (semantic-match collision at build
            # time). Curated fallbacks in templates.py are correct, so force
            # them onto the fallback path.
            if dt not in ("klauzula_niedozwolona", "kara_umowna"):
                if getattr(self, "template_store", None) is not None:
                    text = get_template_real(self.template_store, dt)
            if not text:
                text = get_template_text(dt)
            return template_result(dt, text)

        @fa.post("/resolve_sources")
        async def rs(item: dict):
            """Resolve a SMALL set of cited refs (chunk_id / source_ref) to their
            verbatim corpus chunks. Served from Modal so the generation box
            never opens the local Qdrant for RAG; the payload stays small."""
            refs = (item or {}).get("refs") or []
            out: dict[str, dict] = {}
            for r in refs:
                r = str(r)
                cand = r
                if cand not in self.corpus_map and ":" in r:
                    cand = r.split(":", 1)[1]
                corp = self.corpus_map.get(cand) or self.corpus_map.get(r)
                if corp:
                    out[r] = {k: corp.get(k) for k in RESOLVE_KEEP if corp.get(k) is not None}
            return out

        return fa


@app.cls(
    cpu=4.0,
    memory=8192,
    timeout=900,
    volumes={VOLUME_MOUNT: volume},
    image=image,
    env={
        "HF_HOME": HF_CACHE,
        "HF_TRUST_REMOTE_CODE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    },
)
class RetrievalService(_RetrievalMixin):
    pass


@app.cls(
    gpu="L4",
    cpu=4.0,
    memory=16384,
    timeout=900,
    scaledown_window=300,
    volumes={VOLUME_MOUNT: volume},
    image=image,
    env={
        "HF_HOME": HF_CACHE,
        "HF_TRUST_REMOTE_CODE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    },
)
class RetrievalServiceGpu(_RetrievalMixin):
    """Same API, L4 GPU - for bulk sweeps / interactive agent work."""
