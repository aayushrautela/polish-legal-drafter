"""Bare-metal RAG server: builds the ``legal_corpus`` hybrid index on first
run and serves the retrieval tool API (keyword_search / semantic_search /
chunk_read / get_template) over HTTP.

Provenance: generalized from the rented GPU box that served the eval100
judging (14,197 points, BGE-M3 dense+sparse + bge-reranker-v2-m3 reranking,
27 hand-curated templates via get_template). Runs on any host with the
retrieval dependencies installed; a GPU is strongly recommended for index
builds (embedding 14k chunks on CPU is very slow).

Configuration (all optional, env-driven):
  RAG_REPO        repo root (default: derived from this file's location)
  RAG_CORPUS_DIR  corpus JSONL dir (default: <repo>/dataset/.../sources)
  RAG_QDRANT_PATH index dir (default: <repo>/.rag/qdrant)
  RAG_COLLECTION  collection name (default: legal_corpus)
  RAG_HOST        listen host (default: 0.0.0.0)
  RAG_PORT        listen port (default: 10100)
  RAG_CUDA_ALLOC_CONF  torch allocator tuning (default: expandable_segments:True)

Launch:
  PYTHONPATH=src python3 scripts/rag/server.py
Clients point at it via RETRIEVAL_ENDPOINT / RAG_URL, e.g.
http://127.0.0.1:10100 (same host) or http://<box-ip>:10100 (remote box).
"""
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("rag_server")

# Memory optimization: reduce fragmentation (override with empty string to skip)
if os.environ.get("RAG_CUDA_ALLOC_CONF", "expandable_segments:True"):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = os.environ.get(
        "RAG_CUDA_ALLOC_CONF", "expandable_segments:True")

REPO = Path(os.environ.get(
    "RAG_REPO", Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(REPO / "src"))

from legal_drafter.retrieval.hybrid import QdrantHybridStore, build_hybrid_index, load_corpus
from legal_drafter.retrieval.agentic_tools import keyword_search, semantic_search, chunk_read
from legal_drafter.retrieval.templates import get_template_text, template_result

COLLECTION = os.environ.get("RAG_COLLECTION", "legal_corpus")
HOST = os.environ.get("RAG_HOST", "0.0.0.0")
PORT = int(os.environ.get("RAG_PORT", "10100"))

def _corpus_files() -> list[str]:
    corpus_dir = Path(os.environ.get(
        "RAG_CORPUS_DIR",
        str(REPO / "dataset" / "03_drafting_tasks_sources" / "sources")))
    return sorted(str(p) for p in corpus_dir.glob("*.jsonl"))

def _qdrant_path() -> str:
    return os.environ.get("RAG_QDRANT_PATH", str(REPO / ".rag" / "qdrant"))

def get_store():
    corpus_files = _corpus_files()
    docs = load_corpus(corpus_files)
    log.info("Corpus files: %d", len(corpus_files))

    store = QdrantHybridStore(collection_name=COLLECTION, path=_qdrant_path())
    if store.exists() and store.count >= len(docs) * 0.95:
        log.info("Reusing existing collection (%d points)", store.count)
        return store

    log.info("Building hybrid index (%d docs)...", len(docs))
    t0 = time.time()
    build_hybrid_index(
        corpus_paths=corpus_files,
        qdrant_path=_qdrant_path(),
        collection_name=COLLECTION,
        batch_size=128,
        encode_batch_size=32,
    )
    log.info("Index built in %.1fs (%d points)", time.time() - t0, store.count)
    return store

def main():
    store = get_store()

    from fastapi import FastAPI
    import inspect
    app = FastAPI()
    _tool_lock = threading.Lock()

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

    def _safe(fn, item):
        t0 = time.time()
        log.info("[tool] %s keys=%s", fn.__name__, sorted((item or {}).keys()))
        try:
            return fn(store, **_coerce(fn, item))
        except TypeError as exc:
            req = [p.name for p in inspect.signature(fn).parameters.values()
                   if p.default is inspect.Parameter.empty and p.name != "self"]
            return [{"error": f"{exc}", "hint": f"{fn.__name__} needs {req}; got {sorted((item or {}).keys())}"}]
        except Exception as exc:
            log.error("[tool] %s EXC %s: %s", fn.__name__, type(exc).__name__, exc)
            return [{"error": f"{type(exc).__name__}: {exc}"}]
        finally:
            log.info("[tool] %s done %.0fms", fn.__name__, (time.time() - t0) * 1000)

    @app.get("/health")
    def health():
        return {"status": "ok", "collection": COLLECTION, "points": store.count}

    @app.post("/keyword_search")
    def kw(item: dict):
        with _tool_lock:
            return _safe(keyword_search, item)

    @app.post("/semantic_search")
    def sem(item: dict):
        with _tool_lock:
            return _safe(semantic_search, item)

    @app.post("/chunk_read")
    def cr(item: dict):
        with _tool_lock:
            return _safe(chunk_read, item)

    @app.post("/get_template")
    async def gt(item: dict):
        args = item or {}
        result = get_template_text(args.get("doc_type"))
        return template_result(args.get("doc_type", "other"), result)

    log.info("Starting FastAPI on %s:%d", HOST, PORT)
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")

if __name__ == "__main__":
    main()
