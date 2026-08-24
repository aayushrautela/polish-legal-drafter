"""Agentic RAG retrieval tools exposed to the drafting model.

Implements the A-RAG hierarchical retrieval interface (seq 35):

* ``keyword_search`` - exact lexical match over stored chunk text (no index,
  per A-RAG which pre-indexes nothing for keyword lookups).
* ``semantic_search`` - BGE-M3 dense + sparse hybrid recall with a
  cross-encoder reranker (CPU) over the existing ``.rag/qdrant`` index.
* ``chunk_read`` - full-text fetch of specific chunks by ``chunk_id`` with
  optional same-source context and a context-tracker aware ``exclude_ids``.

All three wrap the existing ``QdrantHybridStore``; no index rebuild is needed.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .embeddings import rerank_scores
from .hybrid import DEFAULT_QDRANT_PATH, QdrantHybridStore
from .templates import get_template_real, get_template_text, template_result

_CACHE: dict[str, list[tuple[int, dict]]] = {}
_ID_INDEX: dict[str, dict[str, int]] = {}
_SOURCE_INDEX: dict[str, dict[str, list[int]]] = {}

# Optional remote backend (Modal retrieval-as-a-service). When set, every tool
# call is dispatched over HTTP instead of running the embedder locally.
_REMOTE = None


def set_remote(client) -> None:
    """Route all tool calls to a RemoteRetrieval (or similar) client."""
    global _REMOTE
    _REMOTE = client


def _scroll_all(store: QdrantHybridStore) -> list[tuple[int, dict]]:
    key = str(store.path)
    if key not in _CACHE:
        out: list[tuple[int, dict]] = []
        nxt = None
        while True:
            pts, nxt = store.client.scroll(
                collection_name=store.collection_name, with_payload=True, limit=2000, offset=nxt
            )
            for p in pts:
                if p.payload:
                    out.append((p.id, p.payload))
            if nxt is None:
                break
        _CACHE[key] = out
        idx: dict[str, int] = {}
        sidx: dict[str, list[int]] = {}
        for pid, pl in out:
            for kk in (pl.get("chunk_id"), pl.get("source_ref")):
                if kk:
                    idx[str(kk)] = pid
            sid = pl.get("source_id") or pl.get("chunk_id")
            if sid:
                sidx.setdefault(str(sid), []).append(pid)
        _ID_INDEX[key] = idx
        _SOURCE_INDEX[key] = sidx
    return _CACHE[key]


def _chunk_text(p: dict) -> str:
    return f"{p.get('title') or ''}\n{p.get('text') or ''}".strip()


def _pass_filters(d: dict, filters: dict | None) -> bool:
    if not filters:
        return True
    if filters.get("legal_area"):
        la = set(d.get("legal_area") or [])
        if not (set(filters["legal_area"]) & la):
            return False
    if filters.get("top_category"):
        if d.get("top_category") not in filters["top_category"]:
            return False
    if filters.get("source_type"):
        if d.get("source_type") not in filters["source_type"]:
            return False
    return True


def _result_dict(p: dict, full: bool) -> dict:
    text = _chunk_text(p)
    if not full and len(text) > 600:
        text = text[:600] + " ..."
    return {
        "chunk_id": p.get("chunk_id") or p.get("source_ref"),
        "source_id": p.get("source_id") or p.get("source_ref"),
        "source_type": p.get("source_type"),
        "top_category": p.get("top_category"),
        "title": p.get("title"),
        "display_address": p.get("display_address"),
        "legal_area": p.get("legal_area"),
        "text": text,
    }


def keyword_search(
    store: QdrantHybridStore,
    keywords: list[str],
    top_k: int = 5,
    filters: dict | None = None,
) -> list[dict]:
    kws = [k.strip().lower() for k in (keywords or []) if k and k.strip()]
    if not kws:
        return []
    scored: list[tuple[int, dict, list[str]]] = []
    for pid, d in _scroll_all(store):
        if not _pass_filters(d, filters):
            continue
        hay = (d.get("text") or "").lower() + " " + (d.get("title") or "").lower()
        score = 0
        hits: list[str] = []
        for k in kws:
            c = hay.count(k)
            if c:
                score += c * len(k)
                hits.append(k)
        if score > 0:
            scored.append((score, d, hits))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [_result_dict(d, full=False) for _, d, _ in scored[:top_k]]


def semantic_search(
    store: QdrantHybridStore,
    query: str,
    top_k: int = 5,
    mode: str = "hybrid",
    filters: dict | None = None,
    candidate_k: int = 20,
) -> list[dict]:
    qfilter = None
    if filters:
        qfilter = {}
        if filters.get("source_type"):
            qfilter["source_types"] = filters["source_type"]
        if filters.get("legal_area"):
            qfilter["legal_area"] = filters["legal_area"]
    cand = store.recall(query, top_k=candidate_k, filters=qfilter or None)
    if not cand:
        return []
    docs = [_chunk_text(p) for _, p in cand]
    ranks = rerank_scores(query, docs)
    order = sorted(range(len(cand)), key=lambda i: ranks[i], reverse=True)
    return [_result_dict(cand[i][1], full=False) for i in order[:top_k]]


def chunk_read(
    store: QdrantHybridStore,
    chunk_ids: list[str],
    with_adjacent: bool = True,
    exclude_ids: list[str] | None = None,
) -> list[dict]:
    key = str(store.path)
    idx = _ID_INDEX.get(key)
    sidx = _SOURCE_INDEX.get(key)
    if idx is None:
        _scroll_all(store)
        idx = _ID_INDEX[key]
        sidx = _SOURCE_INDEX[key]
    exclude = set(exclude_ids or [])
    ids = [str(c) for c in (chunk_ids or [])]
    out: list[dict] = []
    for cid in ids:
        if cid in exclude:
            out.append({"chunk_id": cid, "already_read": True})
            continue
        pid = idx.get(cid)
        if pid is None:
            out.append({"chunk_id": cid, "error": "not found"})
            continue
        pls = store.client.retrieve(collection_name=store.collection_name, ids=[pid], with_payload=True)
        if not pls:
            out.append({"chunk_id": cid, "error": "not found"})
            continue
        payload = pls[0].payload or {}
        out.append(_result_dict(payload, full=True))
        if with_adjacent and sidx:
            sid = payload.get("source_id") or payload.get("chunk_id")
            if sid:
                for apid in sidx.get(str(sid), [])[:2]:
                    if apid == pid:
                        continue
                    apls = store.client.retrieve(
                        collection_name=store.collection_name, ids=[apid], with_payload=True
                    )
                    if apls and apls[0].payload:
                        out.append(_result_dict(apls[0].payload, full=True))
    return out


def get_template(doc_type: str | None = None, query: str | None = None) -> list[dict]:
    """Return the structural contract template for ``doc_type`` as one record.

    ``query`` is optional free-text detail (e.g. "for IT services") appended
    to the base doc_type query to disambiguate between variants.

    Prefers the REAL templates collection (built from the CC-BY-4.0
    legal-templates-multilingual dataset on Modal) when available locally; falls
    back to the small synthetic SCAFFOLD for doc types not yet covered.

    The template is a STRUCTURAL GUIDE (standard sections + required clauses +
    statutory refs + common pitfalls), meant to be used as a structural/style
    guide — NOT copied verbatim. The generator is instructed to adapt clauses to
    the scenario facts and the retrieved real sources, and to cite those sources
    rather than the template.
    """
    dt = (doc_type or "other").strip().lower() or "other"
    text = None
    try:
        local = QdrantHybridStore(collection_name="templates", path=DEFAULT_QDRANT_PATH)
        if local.exists():
            text = get_template_real(local, dt, query=query)
    except Exception:
        text = None
    if not text:
        text = get_template_text(dt)
    return template_result(dt, text)


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "keyword_search",
            "description": (
                "Exact lexical (case-insensitive) match over the legal corpus for known "
                "terms: article numbers (e.g. 'art. 27'), legal terms ('wypowiedzenie'), "
                "statute references ('KC:483'), names. Use short 1-3 word keywords. "
                "Returns chunk snippets with chunk_id; you must chunk_read for full text."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "1-3 word case-insensitive keywords (no phrases)",
                    },
                    "top_k": {"type": "integer", "default": 5, "description": "max results"},
                    "filters": {
                        "type": "object",
                        "properties": {
                            "legal_area": {"type": "array", "items": {"type": "string"}},
                            "top_category": {"type": "array", "items": {"type": "string"}},
                            "source_type": {"type": "array", "items": {"type": "string"}},
                        },
                        "additionalProperties": False,
                    },
                },
                "required": ["keywords"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "semantic_search",
            "description": (
                "Dense/lexical hybrid retrieval for conceptual or paraphrased queries where "
                "exact wording is unknown. Returns chunk snippets with chunk_id; you must "
                "chunk_read for full text."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "natural-language query"},
                    "top_k": {"type": "integer", "default": 5, "description": "max results"},
                    "mode": {
                        "type": "string",
                        "enum": ["hybrid", "dense", "colbert"],
                        "default": "hybrid",
                    },
                    "filters": {
                        "type": "object",
                        "properties": {
                            "legal_area": {"type": "array", "items": {"type": "string"}},
                            "top_category": {"type": "array", "items": {"type": "string"}},
                            "source_type": {"type": "array", "items": {"type": "string"}},
                        },
                        "additionalProperties": False,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "chunk_read",
            "description": (
                "Fetch the FULL text of specific chunks by their chunk_id (from search "
                "results). Always read the full text of a chunk before citing it. Pass "
                "exclude_ids of already-read chunks to avoid redundant reads."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chunk_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "chunk_id values to read in full",
                    },
                    "with_adjacent": {
                        "type": "boolean",
                        "default": True,
                        "description": "also include a little same-source context",
                    },
                    "exclude_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "chunk_ids already read (returned as already_read)",
                    },
                },
                "required": ["chunk_ids"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_template",
            "description": (
                "Fetch the REAL, sourced (CC-BY-4.0) Polish template/guide for the given "
                "doc_type. It provides the authoritative section structure, standard clause "
                "language, and real statutory references for that document type. Use it as the "
                "STRUCTURAL and SUBSTANTIVE SKELETON: follow its section order, reuse its "
                "correct clause wording and statutory citations (they are real, sourced "
                "rules), and adapt every clause to the specific facts and to the REAL sources "
                "you retrieved. Fill or drop any {{placeholders}} as the facts require. Do "
                "not paste the template's generic explanatory guidance verbatim into the "
                "final document; PRESERVE its CC-BY-4.0 attribution. Pass optional 'query' "
                "with extra detail (e.g. 'for IT services with hourly rate') to get a more "
                "specific variant when multiple exist."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_type": {
                        "type": "string",
                        "description": "the doc_type being drafted (e.g. umowa_zlecenia)",
                    },
                    "query": {
                        "type": "string",
                        "description": "optional free-text detail to disambiguate variants (e.g. 'for student hiring')",
                    },
                },
                "required": ["doc_type"],
            },
        },
    },
]


def execute_tool(store: QdrantHybridStore, name: str, arguments: dict) -> list[dict]:
    arguments = arguments or {}
    if _REMOTE is not None:
        fn = getattr(_REMOTE, name, None)
        if fn is not None:
            return fn(**arguments)
        return [{"error": f"unknown tool {name}"}]
    if name == "keyword_search":
        return keyword_search(store, **arguments)
    if name == "semantic_search":
        return semantic_search(store, **arguments)
    if name == "chunk_read":
        return chunk_read(store, **arguments)
    if name == "get_template":
        return get_template(**arguments)
    return [{"error": f"unknown tool {name}"}]
