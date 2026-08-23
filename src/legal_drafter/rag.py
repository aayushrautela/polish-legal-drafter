from __future__ import annotations

import json
import logging
import math
import os
import re
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .retrieval.hybrid import (
    DEFAULT_COLLECTION,
    DEFAULT_QDRANT_PATH,
    QdrantHybridStore,
    rerank_scores,
    rrf_fuse,
)

# Dense hybrid retrieval is opt-in (it downloads BGE-M3 + reranks). Enable with
# LEGAL_DRAFTER_DENSE=1 or by passing dense=True to build_rag_index().
DENSE_DEFAULT = os.environ.get("LEGAL_DRAFTER_DENSE", "0") == "1"

logger = logging.getLogger("legal_drafter.rag")

STOPWORDS = {
    "a", "aby", "albo", "ale", "art", "bez", "by", "być", "co", "czy", "dla", "do", "go", "ich", "i", "jak", "jest", "jeżeli", "lub", "ma", "mi", "mnie", "na", "nad", "nie", "o", "od", "oraz", "po", "pod", "przez", "przy", "się", "są", "ta", "tak", "te", "ten", "to", "u", "w", "we", "z", "za", "ze", "że"
}

SYNONYMS = {
    "wyrzucić": ["wypowiedzenie", "opróżnienie", "eksmisja", "lokator"],
    "wyrzucic": ["wypowiedzenie", "opróżnienie", "eksmisja", "lokator"],
    "kick": ["wypowiedzenie", "opróżnienie", "eksmisja", "lokator"],
    "evict": ["wypowiedzenie", "opróżnienie", "eksmisja", "lokator"],
    "inspection": ["kontrola", "przegląd", "udostępnić", "lokal"],
    "inspekcja": ["kontrola", "przegląd", "udostępnić", "lokal"],
    "surprise": ["bez zgody", "awaria", "policji", "udostępnić"],
    "notice": ["wezwanie", "wypowiedzenie", "termin"],
    "reply": ["odpowiedź", "stanowisko", "zarzut"],
    "payment": ["zapłata", "należność", "odsetki"],
    "rent": ["najem", "lokal", "czynsz", "lokator"],
    "rental": ["najem", "lokal", "czynsz", "lokator"],
    "penalty": ["kara", "umowna", "odsetki", "opóźnienie"],
    "block": ["blokada", "wstrzymanie", "odebranie", "dostęp"],
}

FIELD_WEIGHTS = {
    "chunk_id": 3,
    "source_id": 2,
    "title": 2,
    "display_address": 2,
    "number": 3,
    "article_number": 3,
    "source_type": 1,
    "industry": 2,
    "case_number": 2,
    "legal_bases": 2,
    "court": 1,
    "text": 1,
}

SOURCE_TYPE_BOOST = {
    "statute": 1.18,
    "abusive_clause": 1.18,
    "judgment": 0.72,
}

SOURCE_ID_ALIASES = {
    "uokik_clause_57": "uokik_abusive_clause_57",
    "uokik_clause_77": "uokik_abusive_clause_77",
    "uokik_clause_90": "uokik_abusive_clause_90",
    "uokik_clause_92": "uokik_abusive_clause_92",
    "uokik_clause_300": "uokik_abusive_clause_300",
    "uokik_clause_309": "uokik_abusive_clause_309",
    "uokik_clause_310": "uokik_abusive_clause_310",
}

RULE_RETRIEVALS = {
    "no_contractual_penalty_for_payment_delay": {
        "query": "kara umowna opóźnienie zapłata świadczenie pieniężne odsetki art 481 art 483 art 484",
        "required_source_ids": ["kc_art_481", "kc_art_483", "kc_art_484"],
        "source_types": ["statute", "abusive_clause"],
        "legal_area": ["civil", "contracts", "consumer", "risk_checking"],
    },
    "no_grossly_excessive_or_one_sided_consumer_penalty": {
        "query": "klauzula niedozwolona rażąco wygórowana kara utrata zaliczki rezygnacja odstąpienie 30%",
        "required_source_ids": ["uokik_clause_57", "uokik_clause_90", "kc_art_484", "kc_art_385"],
        "source_types": ["abusive_clause", "statute"],
        "legal_area": ["consumer", "contracts", "risk_checking", "civil"],
    },
    "no_silence_as_acceptance_for_material_changes": {
        "query": "brak odpowiedzi brak sprzeciwu milczenie akceptacja zmiana warunków cena termin program uokik",
        "required_source_ids": ["uokik_clause_300", "kc_art_385"],
        "source_types": ["abusive_clause", "statute"],
        "legal_area": ["consumer", "contracts", "risk_checking", "civil"],
    },
    "no_one_sided_consumer_forum_clause": {
        "query": "sąd właściwy siedziby przedsiębiorcy konsument klauzula niedozwolona uokik",
        "required_source_ids": ["uokik_clause_77", "uokik_clause_92", "uokik_clause_309", "uokik_clause_310", "kc_art_385"],
        "source_types": ["abusive_clause", "statute"],
        "legal_area": ["consumer", "contracts", "risk_checking", "civil"],
    },
    "no_one_sided_immediate_termination_without_settlement": {
        "query": "zlecenie wypowiedzenie w każdym czasie ważny powód rozliczenie wynagrodzenie wykonane czynności świadczenie niespełnione",
        "required_source_ids": ["kc_art_746", "kc_art_491", "kc_art_385"],
        "source_types": ["statute", "abusive_clause"],
        "legal_area": ["civil", "contracts", "consumer", "risk_checking"],
    },
    "no_full_payment_when_access_or_service_blocked": {
        "query": "pełne wynagrodzenie niewykonane świadczenie brak dostępu blokada rozliczenie proporcjonalne nieważność sprzeczność",
        "required_source_ids": ["kc_art_491", "kc_art_746", "kc_art_385"],
        "source_types": ["statute", "abusive_clause"],
        "legal_area": ["civil", "contracts", "consumer", "risk_checking"],
    },
}

SECTION_FILTERS = {
    "2": {
        "prefer_source_types": {"statute", "abusive_clause"},
        "prefer_legal_area": {"civil", "contracts", "consumer", "risk_checking"},
    },
    "3": {
        "prefer_source_types": {"statute", "abusive_clause"},
        "prefer_legal_area": {"civil", "contracts", "consumer", "risk_checking"},
    },
    "4": {
        "prefer_source_types": {"statute", "abusive_clause"},
        "prefer_legal_area": {"civil", "contracts", "consumer", "risk_checking", "tax"},
    },
    "5": {
        "prefer_source_types": {"statute", "abusive_clause"},
        "prefer_legal_area": {"civil", "contracts", "consumer", "risk_checking"},
        "risk_terms": "kara umowna opóźnienie płatność odsetki potrącenie wezwanie do zapłaty rażąco wygórowana",
    },
    "6": {
        "prefer_source_types": {"statute", "abusive_clause"},
        "prefer_legal_area": {"civil", "contracts", "consumer", "risk_checking", "housing"},
        "risk_terms": "wypowiedzenie rozwiązanie natychmiastowe jednostronne zwrot blokada dostęp",
    },
    "7": {
        "prefer_source_types": {"statute", "abusive_clause"},
        "prefer_legal_area": {"civil", "contracts", "consumer", "risk_checking", "civil_procedure"},
        "risk_terms": "właściwość sądu zmiana umowy klauzula niedozwolona nieważność",
    },
}



def normalize(value: str) -> str:
    value = value.lower()
    return value.replace("ą", "a").replace("ć", "c").replace("ę", "e").replace("ł", "l").replace("ń", "n").replace("ó", "o").replace("ś", "s").replace("ź", "z").replace("ż", "z")



def tokenize(value: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9§]+", normalize(value))
    return [token for token in tokens if len(token) > 1 and token not in STOPWORDS]



def expand_query(query: str) -> str:
    additions = []
    lowered = query.lower()
    for key, values in SYNONYMS.items():
        if key in lowered:
            additions.extend(values)
    return " ".join([query, *additions])



def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(as_text(item) for item in value)
    if isinstance(value, dict):
        return " ".join(as_text(item) for item in value.values())
    return str(value)



def preview(value: Any, limit: int = 700) -> str:
    text = re.sub(r"\s+", " ", as_text(value)).strip()
    return text[:limit] + ("..." if len(text) > limit else "")



def source_id(doc: dict[str, Any]) -> str:
    return as_text(doc.get("chunk_id") or doc.get("source_id"))



def document_text(doc: dict[str, Any]) -> str:
    parts = []
    for field, repeat in FIELD_WEIGHTS.items():
        value = as_text(doc.get(field))
        if value:
            parts.extend([value] * repeat)
    parts.extend(as_text(doc.get(field)) for field in ("legal_area", "doc_types", "keywords"))
    return "\n".join(parts)



def load_jsonl(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    from .retrieval.hybrid import normalize_doc

    docs = []
    for path_value in paths:
        path = Path(path_value)
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    docs.append(normalize_doc(json.loads(line)))
    return docs



def build_rag_index(
    paths: Iterable[str | Path],
    *,
    dense: bool | None = None,
    qdrant_path: str | Path | None = None,
    collection_name: str | None = None,
    rebuild: bool = False,
) -> dict[str, Any]:
    if dense is None:
        dense = DENSE_DEFAULT
    docs = load_jsonl(paths)
    term_counts = []
    doc_freq: dict[str, int] = defaultdict(int)
    total_len = 0
    for doc in docs:
        counts = Counter(tokenize(document_text(doc)))
        term_counts.append(counts)
        total_len += sum(counts.values())
        for token in counts:
            doc_freq[token] += 1
    index = {
        "docs": docs,
        "term_counts": term_counts,
        "doc_freq": dict(doc_freq),
        "avg_len": total_len / max(len(docs), 1),
        "by_source_id": {source_id(doc): doc for doc in docs},
        "hybrid": None,
    }
    if dense:
        try:
            store = QdrantHybridStore(
                collection_name=collection_name or DEFAULT_COLLECTION,
                path=qdrant_path or DEFAULT_QDRANT_PATH,
            )
            if rebuild or not store.exists() or store.count != len(docs):
                store.build(docs, rebuild=rebuild)
            index["hybrid"] = store
        except Exception as exc:  # pragma: no cover - heavy deps optional
            logger.warning("Dense hybrid retrieval disabled: %s", exc)
            index["hybrid"] = None
    return index



def score_doc(query_tokens: list[str], counts: Counter[str], doc_freq: dict[str, int], doc_count: int, avg_len: float) -> float:
    score = 0.0
    doc_len = sum(counts.values()) or 1
    k1 = 1.5
    b = 0.75
    for token in query_tokens:
        freq = counts.get(token, 0)
        if freq == 0:
            continue
        token_doc_freq = doc_freq.get(token, 0)
        idf = math.log(1 + (doc_count - token_doc_freq + 0.5) / (token_doc_freq + 0.5))
        score += idf * (freq * (k1 + 1)) / (freq + k1 * (1 - b + b * doc_len / avg_len))
    return score



def resolve_source_id(value: str) -> str:
    if value in SOURCE_ID_ALIASES:
        return SOURCE_ID_ALIASES[value]
    match = re.match(r"^kc_art_(\d+)$", value)
    if match:
        return f"sejm_eli:sejm_eli_du_2024_1061_art_{match.group(1)}"
    return value



def get_source_doc(index: dict[str, Any], source_id_value: str) -> dict[str, Any] | None:
    return index.get("by_source_id", {}).get(resolve_source_id(source_id_value))



def add_required_sources(
    index: dict[str, Any],
    results: list[tuple[float, dict[str, Any]]],
    source_ids: Iterable[str],
    base_score: float,
) -> list[tuple[float, dict[str, Any]]]:
    merged = list(results)
    existing = {source_id(doc) for _, doc in merged}
    for offset, source_id_value in enumerate(source_ids):
        doc = get_source_doc(index, source_id_value)
        if not doc:
            continue
        sid = source_id(doc)
        if sid in existing:
            continue
        merged.append((base_score - (offset * 0.01), doc))
        existing.add(sid)
    return merged



def search_rule_context(
    index: dict[str, Any],
    rule_id: str,
    *,
    top_k: int = 4,
) -> list[tuple[float, dict[str, Any]]]:
    config = RULE_RETRIEVALS.get(rule_id)
    if not config:
        return []
    filters = {
        "source_types": config.get("source_types", []),
        "legal_area": config.get("legal_area", []),
    }
    results = search_rag_index(
        index,
        as_text(config.get("query")),
        top_k=top_k,
        filters=filters,
        candidate_k=max(top_k * 5, 20),
    )
    results = add_required_sources(index, results, config.get("required_source_ids", []), base_score=999.0)
    return dedupe_results(sorted(results, key=lambda item: item[0], reverse=True))[:top_k]



def value_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, list):
        return {normalize(as_text(item)) for item in value if as_text(item)}
    return {normalize(as_text(value))}



def doc_matches_filter(doc: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    if not filters:
        return True
    required_doc_types = {normalize(item) for item in filters.get("doc_types", [])}
    if required_doc_types:
        doc_types = value_set(doc.get("doc_types"))
        if doc_types and doc_types.isdisjoint(required_doc_types):
            return False
    source_types = {normalize(item) for item in filters.get("source_types", [])}
    if source_types:
        source_type = normalize(as_text(doc.get("source_type")))
        if source_type and source_type not in source_types:
            return False
    legal_area = {normalize(item) for item in filters.get("legal_area", [])}
    if legal_area:
        doc_areas = value_set(doc.get("legal_area"))
        if doc_areas and doc_areas.isdisjoint(legal_area):
            return False
    return True



def section_filters(section_number: str, doc_type: str) -> dict[str, Any]:
    config = SECTION_FILTERS.get(section_number, {})
    filters: dict[str, Any] = {"doc_types": [doc_type, "LEGAL_CLAUSE_REVIEW"]}
    if config.get("prefer_source_types"):
        filters["source_types"] = sorted(config["prefer_source_types"])
    if config.get("prefer_legal_area"):
        filters["legal_area"] = sorted(config["prefer_legal_area"])
    return filters



def score_with_boost(score: float, doc: dict[str, Any], section_number: str | None = None) -> float:
    boosted = score * SOURCE_TYPE_BOOST.get(as_text(doc.get("source_type")), 1.0)
    if section_number:
        config = SECTION_FILTERS.get(section_number, {})
        preferred_area = {normalize(item) for item in config.get("prefer_legal_area", set())}
        if preferred_area and not value_set(doc.get("legal_area")).isdisjoint(preferred_area):
            boosted *= 1.08
        preferred_sources = {normalize(item) for item in config.get("prefer_source_types", set())}
        if preferred_sources and normalize(as_text(doc.get("source_type"))) in preferred_sources:
            boosted *= 1.05
    return boosted



def dedupe_results(results: list[tuple[float, dict[str, Any]]]) -> list[tuple[float, dict[str, Any]]]:
    seen = set()
    deduped = []
    for score, doc in results:
        key = source_id(doc)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((score, doc))
    return deduped



def search_rag_index(
    index: dict[str, Any],
    query: str,
    top_k: int = 6,
    filters: dict[str, Any] | None = None,
    section_number: str | None = None,
    candidate_k: int | None = None,
    rerank_top: int = 40,
) -> list[tuple[float, dict[str, Any]]]:
    query_tokens = tokenize(expand_query(query))
    docs = index["docs"]
    term_counts = index["term_counts"]
    doc_freq = index["doc_freq"]
    avg_len = index["avg_len"]
    scored = []
    for doc, counts in zip(docs, term_counts):
        if not doc_matches_filter(doc, filters):
            continue
        score = score_doc(query_tokens, counts, doc_freq, len(docs), avg_len)
        if score > 0:
            scored.append((score_with_boost(score, doc, section_number), doc))
    scored.sort(key=lambda item: item[0], reverse=True)

    hybrid = index.get("hybrid")
    if not hybrid:
        return dedupe_results(scored[: candidate_k or max(top_k * 4, top_k)])[:top_k]

    # --- Dense + sparse recall channel (BGE-M3 + Qdrant, RRF fused) ---
    dense_results: list[tuple[float, dict[str, Any]]] = []
    try:
        dense_results = hybrid.recall(
            query,
            top_k=max(candidate_k or 80, top_k * 4),
            filters=filters,
        )
    except Exception as exc:  # pragma: no cover - heavy deps optional
        logger.warning("Hybrid recall failed, falling back to BM25: %s", exc)
        return dedupe_results(scored[:top_k])

    bm25_keys = [source_id(doc) for _, doc in scored]
    dense_keys = [source_id(doc) for _, doc in dense_results]
    fused = rrf_fuse([bm25_keys, dense_keys])

    candidate_docs: list[dict[str, Any]] = []
    for key in sorted(fused, key=lambda kk: fused[kk], reverse=True)[: max(rerank_top, top_k * 3)]:
        doc = index["by_source_id"].get(key)
        if doc is None:
            for _, d in dense_results:
                if source_id(d) == key:
                    doc = d
                    break
        if doc is not None:
            candidate_docs.append(doc)

    if not candidate_docs:
        return dedupe_results(scored[:top_k])

    # --- Cross-encoder reranking (bge-reranker-v2-m3) ---
    try:
        rerank_texts = [
            f"{as_text(d.get('title'))}\n{as_text(d.get('text'))}" for d in candidate_docs
        ]
        scores = rerank_scores(query, rerank_texts, normalize=True)
        ranked = sorted(zip(candidate_docs, scores), key=lambda item: item[1], reverse=True)
        results = [(float(score), doc) for doc, score in ranked]
    except Exception as exc:  # pragma: no cover - heavy deps optional
        warnings.warn(f"Rerank failed, using RRF fusion scores: {exc}")
        results = [(fused[source_id(doc)], doc) for doc in candidate_docs]

    return dedupe_results(results)[:top_k]



def build_section_rag_query(doc_type: str, doc_title: str, section_number: str, section_title: str, facts: str, section_requirements: dict[str, Any]) -> str:
    config = SECTION_FILTERS.get(section_number, {})
    return "\n".join([
        doc_type,
        doc_title,
        section_title,
        facts,
        as_text(section_requirements.get("topics")),
        as_text(section_requirements.get("must_include_any")),
        as_text(config.get("risk_terms")),
    ])



def triggered_constraint_ids(facts: str, section_number: str, section_title: str) -> list[str]:
    from .legal_constraints import PROHIBITED_RULES, SECTION_HINTS

    context = "\n".join([facts, section_title])
    preferred = set(SECTION_HINTS.get(section_number, []))
    triggered = []
    for rule in PROHIBITED_RULES:
        rule_id = str(rule.get("id", ""))
        if preferred and rule_id in preferred:
            if rule["pattern"].search(context):
                triggered.append(rule_id)
            continue
        if rule["pattern"].search(context):
            triggered.append(rule_id)
    return triggered



def search_section_context(
    index: dict[str, Any],
    *,
    doc_type: str,
    doc_title: str,
    section_number: str,
    section_title: str,
    facts: str,
    section_requirements: dict[str, Any],
    top_k: int = 6,
) -> list[tuple[float, dict[str, Any]]]:
    filters = section_filters(section_number, doc_type)
    drafting_query = build_section_rag_query(doc_type, doc_title, section_number, section_title, facts, section_requirements)
    results = search_rag_index(index, drafting_query, top_k=top_k, filters=filters, section_number=section_number)
    risk_terms = SECTION_FILTERS.get(section_number, {}).get("risk_terms")
    if risk_terms:
        risk_query = "\n".join([doc_type, section_title, facts, as_text(risk_terms), "klauzula niedozwolona nieważność ryzyko prawne"])
        risk_filters = dict(filters)
        risk_filters["source_types"] = ["statute", "abusive_clause"]
        results.extend(search_rag_index(index, risk_query, top_k=max(2, top_k // 2), filters=risk_filters, section_number=section_number))
    for rule_id in triggered_constraint_ids(facts, section_number, section_title):
        results.extend(search_rule_context(index, rule_id, top_k=3))
    results = dedupe_results(sorted(results, key=lambda item: item[0], reverse=True))
    return results[:top_k]



def format_rag_context(results: list[tuple[float, dict[str, Any]]]) -> str:
    if not results:
        return "Brak trafnych źródeł RAG dla tej sekcji."
    lines = []
    for index, (score, doc) in enumerate(results, start=1):
        sid = source_id(doc)
        label = doc.get("display_address") or doc.get("case_number") or as_text(doc.get("case_numbers")) or doc.get("number") or sid
        title = preview(doc.get("title") or doc.get("industry"), 160)
        text = preview(doc.get("text"), 600)
        lines.append(f"[{index}] source_id={sid}; score={score:.2f}; type={doc.get('source_type')}; label={label}; title={title}; url={doc.get('source_url')}\n{text}")
    return "\n\n".join(lines)



def result_summary(results: list[tuple[float, dict[str, Any]]]) -> list[dict[str, Any]]:
    return [
        {
            "score": round(score, 4),
            "chunk_id": doc.get("chunk_id"),
            "source_type": doc.get("source_type"),
            "display_address": doc.get("display_address"),
            "article_number": doc.get("article_number"),
            "case_number": doc.get("case_number"),
            "case_numbers": doc.get("case_numbers"),
            "number": doc.get("number"),
            "title": doc.get("title"),
            "source_url": doc.get("source_url"),
            "text_preview": preview(doc.get("text"), 300),
        }
        for score, doc in results
    ]
