"""Build the BGE-M3 hybrid RAG index on Modal (GPU) and persist it to a Volume.

Run (detached, non-polling):
    modal run --detach modal_build_index.py

The job writes progress to build.log on the Volume and commits the Qdrant
collection. Read status later via:
    modal app logs <app_id>
After completion, download the collection locally (matches the CLI default path):
    mkdir -p .rag/qdrant && modal volume get legal-drafter-rag /qdrant .rag/qdrant
NOTE: the Volume is mounted at /data inside the function, so the CLI path is /qdrant
(not /data/qdrant). `modal volume get` for a DIRECTORY requires the local target dir to
already exist, else it errors with "Is a directory" or nests files under an extra
qdrant/. Pre-creating .rag/qdrant avoids that.

NOTE: Qdrant's embedded (RocksDB) backend needs real local file locking, which a
Modal distributed Volume does not provide reliably. So we build on local ephemeral
disk (/tmp/qdrant) and only COPY the finished collection onto the Volume at the end.
"""

from __future__ import annotations

import logging
import shutil
import sys
import time
from pathlib import Path

import modal

VOLUME_NAME = "legal-drafter-rag"
VOLUME_MOUNT = "/data"
CORPUS_MOUNT = "/corpus"
QDRANT_LOCAL = "/tmp/qdrant"          # local ephemeral disk (RocksDB-safe)
QDRANT_VOLUME = f"{VOLUME_MOUNT}/qdrant"  # final artifact on the Volume
HF_CACHE = f"{VOLUME_MOUNT}/hf_cache"
BUILD_LOG = f"{VOLUME_MOUNT}/build.log"
COLLECTION = "legal_corpus"

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
    )
    .add_local_dir("src/legal_drafter", remote_path="/pkg/legal_drafter")
    .add_local_dir(
        "dataset/03_drafting_tasks_sources/sources",
        remote_path="/corpus",
    )
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

app = modal.App(name="legal-drafter-rag-build")


@app.function(
    image=image,
    gpu="L4",
    timeout=2400,
    volumes={VOLUME_MOUNT: volume},
    env={"HF_HOME": HF_CACHE, "TOKENIZERS_PARALLELISM": "false"},
)
def build_index():
    from legal_drafter.retrieval.hybrid import build_hybrid_index

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(BUILD_LOG), logging.StreamHandler()],
    )
    log = logging.getLogger("rag_build")

    log.info("Starting build: corpus=%s local_qdrant=%s collection=%s", CORPUS_MOUNT, QDRANT_LOCAL, COLLECTION)
    t0 = time.time()
    corpus_files = sorted(Path(CORPUS_MOUNT).glob("*.jsonl"))
    log.info("Found %d corpus files", len(corpus_files))

    # Build on local ephemeral disk (RocksDB-friendly); not on the Volume.
    store = build_hybrid_index(
        corpus_paths=[str(p) for p in corpus_files],
        qdrant_path=QDRANT_LOCAL,
        collection_name=COLLECTION,
        rebuild=True,
        encode_batch_size=64,
        log_every=500,
        batch_size=256,
    )
    try:
        store.close()
    except Exception as exc:  # pragma: no cover - best effort
        log.warning("store.close() error: %s", exc)

    # Copy the finished collection onto the Volume (no active RocksDB there).
    src = Path(QDRANT_LOCAL)
    dst = Path(QDRANT_VOLUME)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    elapsed = time.time() - t0
    log.info("Copied qdrant %s -> %s", src, dst)
    log.info("Build complete in %.1fs. Points=%d", elapsed, getattr(store, "count", -1))

    volume.commit()
    log.info("Volume committed. Status log: %s", BUILD_LOG)


@app.function(
    image=image,
    timeout=600,
    volumes={VOLUME_MOUNT: volume},
)
def tar_qdrant():
    import tarfile
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("rag_tar")
    src = Path(QDRANT_VOLUME)
    dst = Path(f"{VOLUME_MOUNT}/qdrant.tar.gz")
    log.info("Tarring %s -> %s", src, dst)
    with tarfile.open(dst, "w:gz") as tar:
        tar.add(src, arcname="qdrant")
    log.info("Tarred size: %.1f MB", dst.stat().st_size / 1e6)
    volume.commit()
    log.info("Volume committed. Download with: modal volume get %s /qdrant.tar.gz .rag/qdrant.tar.gz", VOLUME_NAME)


@app.local_entrypoint()
def main():
    handle = build_index.spawn()
    print(f"LAUNCHED build_index call_id={handle.object_id}")
    print(f"Read status (do NOT poll): modal app logs <app_id>  (or modal volume get {VOLUME_NAME} {BUILD_LOG} -)")
    print(f"After done, download: mkdir -p .rag/qdrant && modal volume get {VOLUME_NAME} /qdrant .rag/qdrant")
