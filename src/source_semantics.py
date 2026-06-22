from __future__ import annotations

import re
from typing import Any


SOURCE_SEMANTICS: dict[str, dict[str, str]] = {
    "statute": {
        "authority_level": "binding_law",
        "polarity": "normative_rule",
        "legal_effect": "sets_rule_or_condition",
    },
    "abusive_clause": {
        "authority_level": "consumer_blacklist_or_registry",
        "polarity": "negative_example",
        "legal_effect": "similar_clause_may_be_abusive_or_non_binding",
    },
    "judgment": {
        "authority_level": "case_law",
        "polarity": "context_dependent",
        "legal_effect": "requires_holding_or_reasoning_check",
    },
    "form": {
        "authority_level": "official_form_or_template",
        "polarity": "template_example",
        "legal_effect": "format_or_field_guidance_only",
    },
    "template": {
        "authority_level": "template",
        "polarity": "template_example",
        "legal_effect": "style_or_structure_guidance_only",
    },
}


DEFAULT_SEMANTICS = {
    "authority_level": "unknown",
    "polarity": "unknown",
    "legal_effect": "needs_review",
}


NEGATIVE_POLARITIES = {"negative_example"}
POSITIVE_POLARITIES = {"normative_rule"}
WEAK_POLARITIES = {"context_dependent", "template_example", "unknown"}

PROTECTIVE_CLAIM_PATTERNS = [
    r"zgodnie\s+z\s+powszechnie\s+obowiązującymi\s+przepisami",
    r"nie\s+(stanowi|oznacza)\s+akceptacj",
    r"nie\s+zatrzym\w*\s+cał",
    r"niewykorzystan\w*\s+częś\w*[^.]{0,120}(zwrac|zwrot|podlega\s+zwrot)",
    r"odsetk\w*\s+za\s+opóźnienie",
    r"ważn\w*\s+przyczyn",
    r"istotn\w*\s+narus",
    r"rozlicz\w*\s+wykonan\w*\s+usług",
]

RISKY_CLAIM_PATTERNS = [
    r"wyłącznie\s+sąd\w*\s+właściw\w*\s+dla\s+siedzib",
    r"zatrzyma\w*\s+cał\w*\s+zaliczk",
    r"kara\w*\s+umown\w*[^.]{0,160}(30%|opóźn\w*[^.]{0,80}(zapłat|wynagrodz))",
    r"brak\s+(sprzeciwu|odpowiedzi)[^.]{0,160}oznacza\w*\s+akcept",
    r"milczen\w*[^.]{0,160}akcept",
    r"dowoln\w*\s+powod[^.]{0,180}cał\w*\s+wynagrodz",
]


def matches_any(patterns: list[str], value: str) -> bool:
    return any(re.search(pattern, value, re.IGNORECASE) for pattern in patterns)


def claim_stance(claim: dict[str, Any]) -> str:
    text = re.sub(r"\s+", " ", str(claim.get("claim_text") or claim.get("claim") or "").lower())
    if matches_any(PROTECTIVE_CLAIM_PATTERNS, text):
        return "protective"
    if matches_any(RISKY_CLAIM_PATTERNS, text):
        return "risky"
    return "neutral"


def evidence_semantics(evidence: dict[str, Any]) -> dict[str, str]:
    source_type = str(evidence.get("source_type") or "unknown")
    semantics = dict(DEFAULT_SEMANTICS)
    semantics.update(SOURCE_SEMANTICS.get(source_type, {}))
    semantics["source_type"] = source_type
    return semantics


def enrich_evidence_semantics(evidence: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(evidence)
    semantics = evidence_semantics(evidence)
    enriched.update(semantics)
    return enriched


def best_relation_source(verdict: dict[str, Any], evidence: list[dict[str, Any]], relation: str) -> tuple[int | None, dict[str, Any] | None, float]:
    score_key = "entailment" if relation == "entailment" else "contradiction"
    pair_scores = verdict.get("pair_scores") or []
    if not pair_scores:
        preferred_source_id = verdict.get("best_evidence_source_id")
        if preferred_source_id:
            for index, source in enumerate(evidence):
                if source.get("source_id") == preferred_source_id:
                    return index, source, float(verdict.get("confidence") or 0.0)
        if evidence:
            return 0, evidence[0], float(verdict.get("confidence") or 0.0)
        return None, None, 0.0
    best_index = None
    best_score = 0.0
    for index, pair in enumerate(pair_scores):
        if index >= len(evidence):
            continue
        source = evidence[index]
        score = float(pair.get(score_key) or 0.0)
        if relation == "entailment" and source.get("polarity") in NEGATIVE_POLARITIES:
            score += 0.18
        if relation == "entailment" and source.get("polarity") in POSITIVE_POLARITIES:
            score += 0.06
        if score > best_score:
            best_index = index
            best_score = score
    if best_index is None or best_index >= len(evidence):
        return None, None, 0.0
    return best_index, evidence[best_index], best_score


def source_aware_verdict(claim: dict[str, Any], evidence: list[dict[str, Any]], nli_verdict: dict[str, Any]) -> dict[str, Any]:
    relation = str(nli_verdict.get("verdict") or "insufficient")
    stance = claim_stance(claim)
    enriched_evidence = [enrich_evidence_semantics(item) for item in evidence]
    final = dict(nli_verdict)
    final["nli_relation"] = relation
    final["claim_stance"] = stance
    final["evidence_semantics"] = [
        {
            "source_id": item.get("source_id"),
            "source_type": item.get("source_type"),
            "authority_level": item.get("authority_level"),
            "polarity": item.get("polarity"),
            "legal_effect": item.get("legal_effect"),
        }
        for item in enriched_evidence
    ]

    if relation == "supported":
        _, source, score = best_relation_source(nli_verdict, enriched_evidence, "entailment")
        polarity = source.get("polarity") if source else "unknown"
        final["source_polarity"] = polarity
        final["source_authority_level"] = source.get("authority_level") if source else "unknown"
        final["best_evidence_source_id"] = source.get("source_id") if source else nli_verdict.get("best_evidence_source_id")
        final["source_polarity_score"] = round(score, 4)
        if polarity in NEGATIVE_POLARITIES:
            if stance == "protective":
                final["verdict"] = "supported"
                final["final_verdict_reason"] = "protective_claim_is_contrasted_with_negative_example_source"
            else:
                final["verdict"] = "risky"
                final["final_verdict_reason"] = "claim_is_entailed_by_negative_example_source"
        elif polarity in POSITIVE_POLARITIES:
            final["verdict"] = "supported"
            final["final_verdict_reason"] = "claim_is_entailed_by_binding_or_normative_source"
        else:
            final["verdict"] = "needs_review"
            final["final_verdict_reason"] = "claim_matches_context_dependent_or_weak_source"
        return final

    if relation == "contradicted":
        _, source, score = best_relation_source(nli_verdict, enriched_evidence, "contradiction")
        polarity = source.get("polarity") if source else "unknown"
        final["source_polarity"] = polarity
        final["source_authority_level"] = source.get("authority_level") if source else "unknown"
        final["best_evidence_source_id"] = source.get("source_id") if source else nli_verdict.get("best_evidence_source_id")
        final["source_polarity_score"] = round(score, 4)
        if polarity in POSITIVE_POLARITIES:
            final["verdict"] = "contradicted"
            final["final_verdict_reason"] = "claim_contradicts_binding_or_normative_source"
        elif polarity in NEGATIVE_POLARITIES:
            if stance == "risky":
                final["verdict"] = "risky"
                final["final_verdict_reason"] = "risky_claim_related_to_negative_example_source"
            else:
                final["verdict"] = "supported"
                final["final_verdict_reason"] = "claim_is_contrasted_with_negative_example_source"
        else:
            final["verdict"] = "needs_review"
            final["final_verdict_reason"] = "contradiction_against_context_dependent_or_weak_source"
        return final

    final["verdict"] = "insufficient"
    final["source_polarity"] = "none"
    final["source_authority_level"] = "none"
    final["final_verdict_reason"] = "no_conclusive_evidence_relation"
    return final


def map_final_verdict(verdict: str) -> str:
    if verdict == "supported":
        return "supported_or_safe"
    if verdict in {"risky", "contradicted"}:
        return "contradicted_or_risky"
    return "insufficient_evidence"
