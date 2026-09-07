"""Build the ``legal_corpus`` hybrid Qdrant index on a bare-metal GPU box.

Same build as the jobs that produced the frozen 14,197-point index
(BGE-M3 dense+sparse via ``build_hybrid_index``; no ColBERT channel exists
anywhere in the pipeline): embed every ``*.jsonl`` in the corpus dir and
upsert dense + sparse vectors into a local Qdrant collection.

Assumes the retrieval dependencies are installed (torch, FlagEmbedding
with BGE-M3 cached or reachable, qdrant-client) and a GPU is present;
CPU builds work but are very slow. The RAG server
(``scripts/rag/server.py``) reuses an existing complete collection and only
builds when it is missing, so this script is for explicit full rebuilds.

Usage:
    PYTHONPATH=src python3 scripts/rag/build_index.py \
        --corpus dataset/03_drafting_tasks_sources/sources \
        --qdrant-path .rag/qdrant
    # dry count check without building:
    PYTHONPATH=src python3 scripts/rag/build_index.py --no-rebuild
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("rag_build")

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "src"))

from legal_drafter.retrieval.hybrid import (  # noqa: E402
    QdrantHybridStore,
    build_hybrid_index,
    load_corpus,
)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", default=str(
        REPO / "dataset" / "03_drafting_tasks_sources" / "sources"),
        help="corpus dir with *.jsonl source files")
    ap.add_argument("--qdrant-path", default=str(REPO / ".rag" / "qdrant"),
                    help="local Qdrant storage dir")
    ap.add_argument("--collection", default="legal_corpus")
    ap.add_argument("--rebuild", dest="rebuild", action="store_true",
                    default=True, help="force full re-embed (default)")
    ap.add_argument("--no-rebuild", dest="rebuild", action="store_false",
                    help="reuse the collection if it already covers the corpus")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--encode-batch-size", type=int, default=64)
    ap.add_argument("--log-every", type=int, default=500)
    args = ap.parse_args(argv)

    corpus_files = sorted(str(p) for p in Path(args.corpus).glob("*.jsonl"))
    docs = load_corpus(corpus_files)
    log.info("Corpus files: %d, docs: %d", len(corpus_files), len(docs))

    store = QdrantHybridStore(collection_name=args.collection,
                             path=args.qdrant_path)
    if not args.rebuild and store.exists() and store.count >= len(docs) * 0.95:
        log.info("Reusing existing collection (%d points); pass --rebuild "
                 "to force re-embed.", store.count)
        return 0

    t0 = time.time()
    build_hybrid_index(
        corpus_paths=corpus_files,
        qdrant_path=args.qdrant_path,
        collection_name=args.collection,
        rebuild=True,
        encode_batch_size=args.encode_batch_size,
        log_every=args.log_every,
        batch_size=args.batch_size,
    )
    try:
        store.close()
    except Exception:
        pass
    log.info("Index built: %d points in %.1fs", store.count,
             time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
