from __future__ import annotations

import json
import logging
import re
import time
import warnings
from pathlib import Path
from typing import Any, Iterable, Sequence

from .embeddings import encode, rerank_scores

logger = logging.getLogger("legal_drafter.retrieval.hybrid")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QDRANT_PATH = PROJECT_ROOT / ".rag" / "qdrant"
DEFAULT_COLLECTION = "legal_corpus"

# Controlled legal-area vocabulary used by rag.py section filters. Maps the
# corpus `top_category` field onto the legal areas the filters expect.
TOP_CATEGORY_LEGAL_AREA: dict[str | None, set[str]] = {
    "termination": {"civil", "contracts"},
    "payment": {"civil", "contracts"},
    "liability": {"civil", "contracts"},
    "consumer_rights": {"consumer", "contracts"},
    "contract_formation": {"civil", "contracts"},
    "jurisdiction": {"civil", "civil_procedure"},
    "property_or_use_restriction": {"civil", "housing"},
    "penalty_or_interest": {"civil", "contracts"},
    "representation_or_authority": {"civil"},
    "deposit_or_advance": {"housing", "contracts"},
    "complaint_or_notice": {"consumer", "civil"},
    "unilateral_change": {"consumer", "contracts"},
    "withdrawal": {"consumer", "contracts"},
    "price_change": {"consumer", "contracts"},
    "service_quality": {"contracts"},
    "delivery_or_performance_delay": {"contracts"},
    "other": {"civil"},
    None: {"civil"},
}


def normalize_doc(raw: dict[str, Any]) -> dict[str, Any]:
    """Map the recovered corpus schema onto the richer schema rag.py expects.

    The recovered `cleaned_rag_chunks.jsonl` uses ``source_ref``/``top_category``/
    ``risk_or_safe`` while rag.py expects ``chunk_id``/``source_id``/``legal_area``.
    We keep every original field and add the aliases so BM25 dedup, section
    filters and formatting keep working.
    """
    doc = dict(raw)
    ref = raw.get("source_ref") or raw.get("chunk_id") or raw.get("source_id") or ""
    doc.setdefault("chunk_id", ref)
    doc.setdefault("source_id", ref)
    doc.setdefault("display_address", raw.get("title") or ref)
    if not doc.get("legal_area"):
        areas: set[str] = set()
        cat = raw.get("top_category")
        areas |= TOP_CATEGORY_LEGAL_AREA.get(cat, TOP_CATEGORY_LEGAL_AREA[None])
        st = raw.get("source_type")
        if st in ("statute", "abusive_clause", "judgment", "official_form"):
            areas |= {"civil", "contracts", "risk_checking"}
        if st in ("judgment", "official_form"):
            areas |= {"civil_procedure"}
        if raw.get("origin") == "uokik_items":
            areas |= {"consumer"}
        if raw.get("risk_or_safe") in ("risky", "conditional"):
            areas |= {"risk_checking"}
        doc["legal_area"] = sorted(areas)
    return doc


def load_corpus(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    for path_value in paths:
        path = Path(path_value)
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    docs.append(normalize_doc(json.loads(line)))
    return docs


def analyze_corpus(paths: Iterable[str | Path]) -> dict[str, Any]:
    """Profile a corpus so indexing is never a blind box.

    Returns totals, field health, KC article coverage and v1 doc-type coverage.
    """
    import collections

    docs = load_corpus(paths)
    n = len(docs)
    hashes = [d.get("text_hash") for d in docs if d.get("text_hash")]
    lens = [len(d.get("text", "")) for d in docs]
    by_source_type = collections.Counter(d.get("source_type") for d in docs)
    by_origin = collections.Counter(d.get("origin") for d in docs)

    arts: set[int] = set()
    for d in docs:
        m = re.search(r"sejm_eli_du_2024_1061_art_(\d+)", d.get("source_ref", ""))
        if m:
            arts.add(int(m.group(1)))

    def text_of(d: dict) -> str:
        return f"{d.get('title') or ''} {d.get('text') or ''}".lower()

    texts = [text_of(d) for d in docs]
    v1 = {
        "sprzedaż / umowa sprzedaży": ["sprzeda", "rzecz ruchom"],
        "dzieło / umowa o dzieło": ["umowa o dzie", "dzieło"],
        "najem (mieszkaniowy/użytkowy)": ["najem", "najmu"],
        "zlecenie / usługi": ["zlecen", "świadczenie usług"],
        "poręczenie": ["poręczen"],
        "pełnomocnictwo": ["pełnomoc"],
        "NDA / poufność": ["poufno", "tajemnic"],
        "wezwanie do zapłaty": ["wezwanie do zapłaty", "wezwanie o zapłat"],
        "reklamacja": ["reklamacj"],
        "odstąpienie (konsument 14 dni)": ["odstąpienie od umowy", "prawach konsumenta"],
        "kaucja": ["kaucj"],
    }
    v1_coverage = {name: sum(1 for t in texts if any(k in t for k in kws)) for name, kws in v1.items()}
    consumer_act_present = any("prawach konsumenta" in t for t in texts)

    return {
        "total": n,
        "unique_text_hash": len(set(hashes)),
        "duplicate_text_hash": len(hashes) - len(set(hashes)),
        "avg_chars": (sum(lens) // n) if n else 0,
        "min_chars": min(lens) if lens else 0,
        "max_chars": max(lens) if lens else 0,
        "by_source_type": dict(by_source_type),
        "by_origin": dict(by_origin),
        "kc_articles_present": len(arts),
        "kc_article_range": [min(arts), max(arts)] if arts else None,
        "v1_coverage": v1_coverage,
        "consumer_act_present": consumer_act_present,
        "gaps": [name for name, c in v1_coverage.items() if c == 0],
    }


def _qdrant_filter(filters: dict[str, Any] | None):
    from qdrant_client import models

    if not filters:
        return None
    must: list[Any] = []
    if filters.get("source_types"):
        must.append(
            models.FieldCondition(
                key="source_type",
                match=models.MatchAny(any=list(filters["source_types"])),
            )
        )
    if filters.get("legal_area"):
        must.append(
            models.FieldCondition(
                key="legal_area",
                match=models.MatchAny(any=list(filters["legal_area"])),
            )
        )
    if filters.get("doc_types"):
        must.append(
            models.FieldCondition(
                key="doc_types",
                match=models.MatchAny(any=list(filters["doc_types"])),
            )
        )
    return models.Filter(must=must) if must else None


def rrf_fuse(ranked_keys: Sequence[list[str]], k: int = 60) -> dict[str, float]:
    """Reciprocal Rank Fusion across ranked id lists.

    Each input is an ordered list of document keys (best first). Returns a dict
    key -> fused score (higher = better).
    """
    fused: dict[str, float] = {}
    for ranked in ranked_keys:
        for rank, key in enumerate(ranked):
            fused[key] = fused.get(key, 0.0) + 1.0 / (k + rank + 1)
    return fused


LEGAL_REF_PATTERNS = [
    (re.compile(r"art\.?\s*(\d+)\s*(?:§\s*(\d+))?\s*k\.?c\.?", re.I), "kc"),
    (re.compile(r"art\.?\s*(\d+)\s*(?:§\s*(\d+))?\s*k\.?p\.?c\.?", re.I), "kpc"),
    (re.compile(r"u\.?o\.?p\.?l\.?", re.I), "ustawa_o_ochronie_praw_lokatorow"),
    (re.compile(r"ustaw(?:a|y)?\s+o\s+ochronie\s+praw\s+lokator", re.I), "ustawa_o_ochronie_praw_lokatorow"),
    (re.compile(r"ustaw(?:a|y)?\s+o\s+prawach\s+konsumenta", re.I), "ustawa_o_prawach_konsumenta"),
    (re.compile(r"art\.?\s*(\d+)\s*ustaw", re.I), "ustawa_art"),
]


def extract_legal_refs(text: str) -> list[dict[str, Any]]:
    """Extract lightweight legal references (article / act mentions) from text."""
    refs: list[dict[str, Any]] = []
    for pattern, kind in LEGAL_REF_PATTERNS:
        for match in pattern.finditer(text or ""):
            ref: dict[str, Any] = {"kind": kind}
            if match.groups() and match.group(1):
                ref["article"] = int(match.group(1))
                if len(match.groups()) > 1 and match.group(2):
                    ref["paragraph"] = int(match.group(2))
            ref["text"] = match.group(0)
            refs.append(ref)
    return refs


def grounding_hits(query: str, results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Count how many legal refs mentioned in the query are grounded in retrieved docs.

    Returns {"refs": [...], "grounded": int, "total": int, "ratio": float}.
    """
    refs = extract_legal_refs(query)
    if not refs:
        return {"refs": [], "grounded": 0, "total": 0, "ratio": 0.0}
    grounded = 0
    for ref in refs:
        hay = " ".join(
            f"{r.get('source_ref', '')} {r.get('title', '')} {r.get('text', '')}"
            for r in results
        )
        if ref.get("article"):
            token = f"art. {ref['article']}"
            if token in hay or f"art.{ref['article']}" in hay:
                grounded += 1
        else:
            if ref["kind"] in hay:
                grounded += 1
    return {
        "refs": refs,
        "grounded": grounded,
        "total": len(refs),
        "ratio": grounded / len(refs),
    }


class QdrantHybridStore:
    """Embedded Qdrant store with BGE-M3 dense + sparse vectors and RRF recall."""

    # Embedded Qdrant allows only ONE client per storage folder. Cache clients
    # by path so multiple stores/collections in the same folder share one
    # low-level client instead of racing for the RocksDB lock.
    _CLIENTS: dict[str, "QdrantClient"] = {}

    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION,
        path: str | Path = DEFAULT_QDRANT_PATH,
        client: "QdrantClient | None" = None,
    ) -> None:
        self.collection_name = collection_name
        self.path = Path(path)
        self._client = client
        self._count: int | None = None

    @property
    def client(self):
        if self._client is None:
            from qdrant_client import QdrantClient

            key = str(self.path)
            cached = QdrantHybridStore._CLIENTS.get(key)
            if cached is None:
                self.path.mkdir(parents=True, exist_ok=True)
                cached = QdrantClient(path=key)
                QdrantHybridStore._CLIENTS[key] = cached
            self._client = cached
        return self._client

    def exists(self) -> bool:
        return self.client.collection_exists(self.collection_name)

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    @property
    def count(self) -> int:
        if self._count is None:
            self._count = self.client.count(self.collection_name).count if self.exists() else 0
        return self._count

    def ensure_collection(self) -> None:
        from qdrant_client import models

        if self.client.collection_exists(self.collection_name):
            return
        self.client.create_collection(
            self.collection_name,
            vectors_config={
                "dense": models.VectorParams(size=1024, distance=models.Distance.COSINE)
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(index=models.SparseIndexParams(on_disk=False))
            },
        )

    def build(
        self,
        docs: Sequence[dict[str, Any]],
        batch_size: int = 16,
        rebuild: bool = False,
        encode_batch_size: int = 32,
        log_every: int = 500,
    ) -> int:
        from qdrant_client import models

        if self.exists() and rebuild:
            logger.info("Deleting existing collection '%s' (rebuild=True)", self.collection_name)
            self.client.delete_collection(self.collection_name)
            self._count = None
        self.ensure_collection()
        docs = [normalize_doc(d) for d in docs]
        n = len(docs)
        failures = 0
        start_time = time.time()
        logger.info(
            "Embedding %d docs into collection '%s' (upsert_batch=%d, encode_batch=%d)",
            n,
            self.collection_name,
            batch_size,
            encode_batch_size,
        )
        points: list[models.PointStruct] = []
        done = 0
        last_log = 0
        for i in range(0, n, encode_batch_size):
            chunk = docs[i : i + encode_batch_size]
            texts = [f"{d.get('title') or ''}\n{d.get('text') or ''}".strip() for d in chunk]
            try:
                dense, sparse = encode(texts)
            except Exception:
                failures += len(chunk)
                logger.exception("Failed to embed docs %d-%d", i, i + len(chunk) - 1)
                continue
            for j, d in enumerate(chunk):
                points.append(
                    models.PointStruct(
                        id=i + j,
                        payload=d,
                        vector={"dense": dense[j], "sparse": sparse[j]},
                    )
                )
            done += len(chunk)
            if len(points) >= batch_size:
                self.client.upsert(self.collection_name, points=points)
                points = []
            if log_every and done // log_every > last_log:
                last_log = done // log_every
                elapsed = time.time() - start_time
                logger.info("Progress %d/%d (%.1f%%) in %.1fs", done, n, 100.0 * done / n, elapsed)
        if points:
            self.client.upsert(self.collection_name, points=points)
        self._count = self.client.count(self.collection_name).count
        elapsed = time.time() - start_time
        if failures:
            logger.warning(
                "Built collection '%s': %d points in %.1fs (%d embedding failures)",
                self.collection_name,
                self._count,
                elapsed,
                failures,
            )
        else:
            logger.info(
                "Built collection '%s': %d points in %.1fs", self.collection_name, self._count, elapsed
            )
        return self._count

    def recall(
        self,
        query: str,
        top_k: int = 20,
        candidate_k: int = 80,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[float, dict[str, Any]]]:
        from qdrant_client import models

        dense, sparse = encode([query])
        qfilter = _qdrant_filter(filters)
        prefetch = [
            models.Prefetch(query=sparse[0], using="sparse", limit=candidate_k),
            models.Prefetch(query=dense[0], using="dense", limit=candidate_k),
        ]
        response = self.client.query_points(
            self.collection_name,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            with_payload=True,
            limit=top_k,
            query_filter=qfilter,
        )
        logger.debug("Hybrid recall returned %d hits", len(response.points))
        return [(float(h.score), h.payload) for h in response.points]


def build_hybrid_index(
    corpus_paths: Iterable[str | Path],
    qdrant_path: str | Path = DEFAULT_QDRANT_PATH,
    collection_name: str = DEFAULT_COLLECTION,
    rebuild: bool = False,
    batch_size: int = 16,
    encode_batch_size: int = 32,
    log_every: int = 500,
) -> QdrantHybridStore:
    """Load a corpus and build (or reuse) the Qdrant hybrid index on disk."""
    store = QdrantHybridStore(collection_name=collection_name, path=qdrant_path)
    docs = load_corpus(corpus_paths)
    if rebuild or not store.exists() or store.count != len(docs):
        logger.info(
            "Building hybrid index: %d docs -> %s @ %s", len(docs), collection_name, qdrant_path
        )
        store.build(
            docs,
            batch_size=batch_size,
            rebuild=rebuild,
            encode_batch_size=encode_batch_size,
            log_every=log_every,
        )
    else:
        logger.info(
            "Reusing existing collection '%s' (%d points) — pass rebuild=True to force re-embed.",
            collection_name,
            store.count,
        )
    return store
