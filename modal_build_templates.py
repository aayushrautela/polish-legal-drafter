"""Build the real ``templates`` Qdrant collection on Modal (GPU) and merge it
into the existing ``legal-drafter-rag`` Volume alongside ``legal_corpus``.

The templates come from the CC-BY-4.0 ``Anyone5559/legal-templates-multilingual``
dataset (Polish rows), prepared as ``/data/templates_pl.jsonl`` on the volume
(see convert step below). Each row becomes one point whose ``text`` is the
concatenation of the substantive guidance fields (key_elements, legal_requirements,
how_to_fill, common_mistisks, ...).

Unlike modal_build_index.py this does NOT replace the whole qdrant dir. It builds
the new collection on ephemeral ``/tmp/qdrant`` and copies ONLY
``collections/templates`` into the volume's qdrant, leaving ``legal_corpus``
untouched.

Local prep (convert the pulled parquet -> JSONL, then upload):
    .venv/bin/python - <<'PY'
    import pandas as pd, json
    df = pd.read_parquet("/tmp/opencode/lm/templates.parquet")
    pl = df[df["locale"] == "pl"]
    rows = []
    for _, r in pl.iterrows():
        d = {k: (None if pd.isna(r[k]) else r[k]) for k in
             ["title","category","subcategory","url_slug","source_url",
              "primary_statute","locale","what_is","when_needed","key_elements",
              "how_to_fill","legal_requirements","common_mistakes","faq_json"]}
        rows.append(d)
    with open("/tmp/opencode/lm/templates_pl.jsonl","w",encoding="utf-8") as f:
        for d in rows:
            f.write(json.dumps(d, ensure_ascii=False)+"\\n")
    print("wrote", len(rows), "rows")
    PY
    modal volume put legal-drafter-rag /tmp/opencode/lm/templates_pl.jsonl /templates_pl.jsonl

Run (detached, non-polling):
    modal run --detach modal_build_templates.py
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path

import modal

VOLUME_NAME = "legal-drafter-rag"
VOLUME_MOUNT = "/data"
QDRANT_LOCAL = "/tmp/qdrant"           # local ephemeral disk (RocksDB-safe)
QDRANT_VOLUME = f"{VOLUME_MOUNT}/qdrant"  # final artifact on the Volume
HF_CACHE = f"{VOLUME_MOUNT}/hf_cache"
TEMPLATES_JSONL = f"{VOLUME_MOUNT}/templates_pl.jsonl"
COLLECTION = "templates"

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
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

app = modal.App(name="legal-drafter-templates-build")


@app.function(
    image=image,
    gpu="L40S",
    timeout=2400,
    volumes={VOLUME_MOUNT: volume},
    env={
        "HF_HOME": HF_CACHE,
        "HF_TRUST_REMOTE_CODE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    },
)
def build_templates_index():
    from legal_drafter.retrieval.hybrid import QdrantHybridStore
    from legal_drafter.retrieval.templates import build_template_text

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    log = logging.getLogger("templates_build")

    src = Path(TEMPLATES_JSONL)
    if not src.exists():
        raise FileNotFoundError(f"expected prepared JSONL at {src}")

    docs: list[dict] = []
    with src.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = build_template_text(row)
            if not text.strip():
                continue
            docs.append(
                {
                    "title": row.get("title") or "",
                    "text": text,
                    "source_id": row.get("url_slug") or f"tpl_{len(docs)}",
                    "source_type": "contract_template",
                    "doc_type": row.get("doc_type") or "",
                    "category": row.get("category"),
                    "url_slug": row.get("url_slug"),
                    "source_url": row.get("source_url"),
                    "primary_statute": row.get("primary_statute"),
                    "locale": row.get("locale") or "pl",
                }
            )
    log.info("Loaded %d template docs", len(docs))

    store = QdrantHybridStore(collection_name=COLLECTION, path=QDRANT_LOCAL)
    t0 = time.time()
    store.build(docs, batch_size=256, rebuild=True, encode_batch_size=16, log_every=50)
    try:
        store.close()
    except Exception as exc:  # pragma: no cover - best effort
        log.warning("store.close() error: %s", exc)

    # Properly MERGE the new collection into the existing volume qdrant so that
    # BOTH `legal_corpus` and `templates` are registered. Copying just a
    # collection subdir does NOT update Qdrant's root metadata, so the existing
    # `legal_corpus` qdrant would never "see" the new collection. Instead we copy
    # the whole existing qdrant to ephemeral disk, register `templates` into it
    # via the Qdrant API (create_collection + upsert), then replace the volume
    # qdrant entirely (same pattern as modal_build_index.build_index).
    import os

    from qdrant_client import QdrantClient, models

    merged_qdrant = "/tmp/merged_qdrant"
    if os.path.exists(merged_qdrant):
        shutil.rmtree(merged_qdrant)
    shutil.copytree(QDRANT_VOLUME, merged_qdrant)

    src_client = QdrantClient(path=QDRANT_LOCAL)
    tgt_client = QdrantClient(path=merged_qdrant)
    # Remove any stale (unregistered) templates dir from the copied qdrant first.
    stale = os.path.join(merged_qdrant, "collection", COLLECTION)
    if os.path.exists(stale):
        shutil.rmtree(stale)
    tgt_client.create_collection(
        COLLECTION,
        vectors_config={"dense": models.VectorParams(size=1024, distance=models.Distance.COSINE)},
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(index=models.SparseIndexParams(on_disk=False))
        },
    )
    offset = None
    copied = 0
    while True:
        pts, offset = src_client.scroll(COLLECTION, with_vectors=True, limit=500, offset=offset)
        if pts:
            tgt_client.upsert(COLLECTION, points=pts)
            copied += len(pts)
        if offset is None:
            break
    log.info("Registered %d template points into merged qdrant", copied)
    tgt_client.close()
    src_client.close()

    if os.path.exists(QDRANT_VOLUME):
        shutil.rmtree(QDRANT_VOLUME)
    shutil.copytree(merged_qdrant, QDRANT_VOLUME)
    elapsed = time.time() - t0
    log.info("Replaced volume qdrant with merged (%.1fs)", elapsed)

    volume.commit()
    log.info("Volume committed. templates collection ready.")


@app.local_entrypoint()
def main():
    handle = build_templates_index.spawn()
    print(f"LAUNCHED build_templates_index call_id={handle.object_id}")
    print("Do NOT poll. Check later via: modal app logs legal-drafter-templates-build")
    print("Then redeploy the retrieval service: modal deploy modal_retrieval.py")
