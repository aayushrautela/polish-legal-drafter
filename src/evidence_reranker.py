from __future__ import annotations

import re
from typing import Any

from .rag import preview, search_rag_index, source_id
from .source_semantics import enrich_evidence_semantics


DEFAULT_RAG_PATHS = [
    "data/rag/sejm_eli_chunks.jsonl",
    "data/rag/risk_chunks.jsonl",
]
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


def claim_query(claim: dict[str, Any]) -> str:
    return "\n".join(
        part
        for part in [
            str(claim.get("claim_text") or ""),
            str(claim.get("section_title") or ""),
            str(claim.get("case_id") or ""),
        ]
        if part
    )


def cited_sources(index: dict[str, Any], claim: dict[str, Any]) -> list[tuple[float, dict[str, Any]]]:
    docs = []
    by_source_id = index.get("by_source_id", {})
    for offset, sid in enumerate(claim.get("source_ids_from_clause") or []):
        doc = by_source_id.get(str(sid))
        if doc:
            docs.append((1000.0 - offset, doc))
    return docs


def lexical_overlap_score(claim_text: str, evidence_text: str) -> float:
    claim_tokens = {token for token in re.findall(r"[a-ząćęłńóśźż0-9]+", claim_text.lower()) if len(token) > 2}
    evidence_tokens = {token for token in re.findall(r"[a-ząćęłńóśźż0-9]+", evidence_text.lower()) if len(token) > 2}
    if not claim_tokens or not evidence_tokens:
        return 0.0
    return len(claim_tokens & evidence_tokens) / len(claim_tokens)


def evidence_text(doc: dict[str, Any]) -> str:
    return "\n".join(
        str(part)
        for part in [
            doc.get("title"),
            doc.get("display_address"),
            doc.get("number"),
            doc.get("industry"),
            doc.get("text"),
        ]
        if part
    )


def evidence_record(doc: dict[str, Any], *, score: float, lexical_overlap: float | None = None, model_score: float | None = None, backend: str = "lexical") -> dict[str, Any]:
    record = {
        "source_id": source_id(doc),
        "source_type": doc.get("source_type"),
        "title": doc.get("title"),
        "display_address": doc.get("display_address"),
        "number": doc.get("number"),
        "source_url": doc.get("source_url"),
        "score": round(score, 4),
        "rerank_backend": backend,
        "text": preview(doc.get("text"), 1000),
    }
    if lexical_overlap is not None:
        record["lexical_overlap"] = round(lexical_overlap, 4)
    if model_score is not None:
        record["model_score"] = round(model_score, 6)
    return enrich_evidence_semantics(record)


def dedupe_candidates(candidates: list[tuple[float, dict[str, Any]]]) -> list[tuple[float, dict[str, Any]]]:
    seen: set[str] = set()
    deduped = []
    for score, doc in candidates:
        sid = source_id(doc)
        if sid in seen:
            continue
        seen.add(sid)
        deduped.append((score, doc))
    return deduped


def rerank_evidence_fallback(claim: dict[str, Any], candidates: list[tuple[float, dict[str, Any]]], *, top_k: int = 5) -> list[dict[str, Any]]:
    claim_text = str(claim.get("claim_text") or "")
    ranked = []
    for base_score, doc in dedupe_candidates(candidates):
        overlap = lexical_overlap_score(claim_text, evidence_text(doc))
        score = float(base_score) + (overlap * 25.0)
        ranked.append((score, overlap, doc))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [evidence_record(doc, score=score, lexical_overlap=overlap, backend="lexical") for score, overlap, doc in ranked[:top_k]]


class CrossEncoderEvidenceReranker:
    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL, *, device: str | None = None, max_length: int = 512, batch_size: int = 8) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def scores(self, query: str, passages: list[str]) -> list[float]:
        values: list[float] = []
        with self.torch.inference_mode():
            for start in range(0, len(passages), self.batch_size):
                batch = passages[start:start + self.batch_size]
                encoded = self.tokenizer(
                    [query] * len(batch),
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                logits = self.model(**encoded).logits.detach().float().cpu()
                if logits.ndim == 2 and logits.shape[1] > 1:
                    batch_scores = logits[:, -1]
                else:
                    batch_scores = logits.reshape(-1)
                values.extend(float(value) for value in batch_scores.tolist())
        return values

    def rerank(self, claim: dict[str, Any], candidates: list[tuple[float, dict[str, Any]]], *, top_k: int = 5) -> list[dict[str, Any]]:
        deduped = dedupe_candidates(candidates)
        query = claim_query(claim)
        passages = [evidence_text(doc) for _, doc in deduped]
        model_scores = self.scores(query, passages) if passages else []
        ranked = []
        for (base_score, doc), model_score in zip(deduped, model_scores):
            overlap = lexical_overlap_score(str(claim.get("claim_text") or ""), evidence_text(doc))
            score = float(model_score) + min(float(base_score), 1000.0) / 1000.0
            ranked.append((score, overlap, model_score, doc))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [
            evidence_record(doc, score=score, lexical_overlap=overlap, model_score=model_score, backend=self.model_name)
            for score, overlap, model_score, doc in ranked[:top_k]
        ]


def select_evidence(
    index: dict[str, Any],
    claim: dict[str, Any],
    *,
    candidate_k: int = 30,
    top_k: int = 5,
    reranker: CrossEncoderEvidenceReranker | None = None,
) -> list[dict[str, Any]]:
    candidates = cited_sources(index, claim)
    candidates.extend(search_rag_index(index, claim_query(claim), top_k=candidate_k, candidate_k=candidate_k * 2))
    if reranker:
        return reranker.rerank(claim, candidates, top_k=top_k)
    return rerank_evidence_fallback(claim, candidates, top_k=top_k)
