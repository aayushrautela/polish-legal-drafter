from __future__ import annotations

import re
from typing import Any

from .legal_constraints import constraint_violation, constraints_from_rag_debug

LEGAL_REFERENCE_RE = re.compile(r"\b(art\.|kodeks|ustaw|uokik|rejestr|sąd|nieważn|niedozwolon|odsetk|kara umowna|wypowiedzenie|rozwiązanie)\b", re.IGNORECASE)
NEGATION_RE = re.compile(r"\b(nie|bez|zakaz|niedopuszczal|niezgodn|ryzyk|nie\s+może|nie\s+moze|nie\s+podlega)\b", re.IGNORECASE)

RISKY_PATTERNS = {
    "monetary_contractual_penalty": {
        "pattern": re.compile(r"kara\w*\s+umown\w*[^.]{0,220}((za|z\s+tytułu|w\s+przypadku)\s+[^.]{0,90}(opóźn|zwłok)[^.]{0,90}(płatno|zapłat|wynagrodzen|należno)|opóźn\w*[^.]{0,140}(płatno|zapłat|wynagrodzen|należno)|zwłok\w*[^.]{0,140}(płatno|zapłat|wynagrodzen|należno))", re.IGNORECASE),
        "safe": re.compile(r"(nie\s+(przewiduje|nalicza|stosuje|zastrzega)|kara\w*\s+umown\w*[^.]{0,120}nie\s+dotyczy|zamiast\s+kary[^.]{0,80}odsetk)", re.IGNORECASE),
        "severity": "high",
    },
    "full_payment_during_block": {
        "pattern": re.compile(r"(pełn\w*\s+(wynagrodzen|czynsz)[^.]{0,220}(blokad|odebran|wstrzym|zablok|brak\s+dostęp|niewykonan)|blokad\w*[^.]{0,220}pełn\w*\s+(wynagrodzen|czynsz)|zachow\w*[^.]{0,80}(całość|pełn\w*)\s+wynagrodzen[^.]{0,160}niewykonan)", re.IGNORECASE),
        "safe": re.compile(r"(nie\s+(może|moze|zachowuje|przysługuje)|proporcjonaln\w*\s+rozlicz|obniż|zwrot\w*[^.]{0,100}niewykonan|niewykorzystan\w*\s+częś)", re.IGNORECASE),
        "severity": "high",
    },
    "unilateral_immediate_termination": {
        "pattern": re.compile(r"((natychmiastow\w*\s+(rozwiąz|wypowied)|bez\s+zachowania\s+(terminu|okresu)|bez\s+okresu\s+wypowiedzenia)[^.]{0,180}(dowoln\w*\s+powod|bez\s+podania\s+przyczyn|jakiejkolwiek\s+przyczyn)?|zachow\w*[^.]{0,80}(całość|pełn\w*)\s+wynagrodzen[^.]{0,160}niewykonan)", re.IGNORECASE),
        "safe": re.compile(r"(nie\s+(może|moze|uprawnia|pozwala)|ważn\w*\s+przyczyn|istotn\w*\s+narus|proporcjonaln\w*\s+rozlicz|zwrot\w*[^.]{0,100}niewykonan)", re.IGNORECASE),
        "severity": "high",
    },
    "wrong_court_consumer_risk": {
        "pattern": re.compile(r"sąd\w*\s+właściw\w*[^.]{0,160}(siedzib|zleceniodawc|przedsiębiorc|wykonawc)", re.IGNORECASE),
        "safe": re.compile(r"(według\s+przepis|zgodnie\s+z\s+przepisami|nie\s+narzuca|nie\s+ogranicza|bez\s+jednostronnego)", re.IGNORECASE),
        "severity": "medium",
    },
    "surprise_access": {
        "pattern": re.compile(r"(bez\s+(zgody|uprzedzenia)|w\s+każdym\s+czasie)[^.]{0,160}(wejść|wejsc|kontrol|lokal|sprzęt|sprzet)", re.IGNORECASE),
        "safe": re.compile(r"(nie\s+(może|moze)|po\s+uprzednim|za\s+zgodą|uzgodnionym\s+termin)", re.IGNORECASE),
        "severity": "medium",
    },
    "silence_acceptance_material_change": {
        "pattern": re.compile(r"(brak\s+(sprzeciwu|odpowiedzi)|milczen\w*|3\s+dni)[^.]{0,160}(oznacz|traktowan|uznan|równoznaczn|akcept)", re.IGNORECASE),
        "safe": re.compile(r"((brak\s+(sprzeciwu|odpowiedzi)|milczen\w*)[^.]{0,120}nie\s+(oznacza|stanowi|jest|będzie|może|moze)|nie\s+(może|moze|mogą|moga|oznacza|stanowi|jest|będzie)[^.]{0,120}(akcept|zgod))", re.IGNORECASE),
        "severity": "high",
    },
}

RISKY_SOURCE_HINTS = {
    "monetary_contractual_penalty": ["kc_art_483", "kc_art_484", "uokik"],
    "full_payment_during_block": ["kc_art_3531", "kc_art_58", "uokik"],
    "unilateral_immediate_termination": ["kc", "uokik"],
    "wrong_court_consumer_risk": ["uokik", "kpc"],
    "surprise_access": ["uokik", "tenant_protection"],
    "silence_acceptance_material_change": ["uokik"],
}

CONTRADICTION_PAIRS = [
    (
        re.compile(r"nie\s+może[^.]{0,180}(blokow|odebra|wstrzym)", re.IGNORECASE),
        re.compile(r"ma\s+prawo[^.]{0,180}(blokow|odebra|wstrzym)|może[^.]{0,180}(blokow|odebra|wstrzym)", re.IGNORECASE),
        "blocking_access_conflict",
    ),
    (
        re.compile(r"kara\w*\s+umown\w*[^.]{0,180}(rażąco|wygórowan|zmniejszen|miarkowan|ryzyk)", re.IGNORECASE),
        re.compile(r"kara\w*\s+umown\w*[^.]{0,180}(nieograniczon|odstrasz|każde\s+opóźnienie|samo\s+opóźnienie)", re.IGNORECASE),
        "penalty_clause_conflict",
    ),
]


def clause_source_ids(clause: dict[str, Any]) -> list[str]:
    values = clause.get("source_ids", [])
    if not isinstance(values, list):
        return []
    return [str(value) for value in values if isinstance(value, str) and value.strip()]


def clause_records(sections: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for section_number, section in sections.items():
        for clause in section.get("clauses", []):
            if not isinstance(clause, dict):
                continue
            text = str(clause.get("text", ""))
            records.append({
                "section_number": section_number,
                "clause_id": clause.get("id"),
                "text": text,
                "source_ids": clause_source_ids(clause),
            })
    return records


def is_safe_negation(text: str, pattern_config: dict[str, Any]) -> bool:
    safe = pattern_config.get("safe")
    if safe and safe.search(text):
        return True
    return False


def validate_sources(sections: dict[str, dict[str, Any]], allowed_source_ids: set[str]) -> list[dict[str, Any]]:
    warnings = []
    for record in clause_records(sections):
        text = record["text"]
        source_ids = record["source_ids"]
        if LEGAL_REFERENCE_RE.search(text) and not source_ids:
            warnings.append({
                "type": "legal_claim_without_source",
                "severity": "medium",
                "section_number": record["section_number"],
                "clause_id": record["clause_id"],
                "message": "Legal/risk claim has no source_ids.",
                "text_preview": text[:400],
            })
        for source_id in source_ids:
            if source_id not in allowed_source_ids:
                warnings.append({
                    "type": "unknown_source_id",
                    "severity": "high",
                    "section_number": record["section_number"],
                    "clause_id": record["clause_id"],
                    "message": f"Clause cites a source_id not present in retrieved RAG context: {source_id}",
                    "text_preview": text[:400],
                })
    return warnings


def validate_clause_risks(sections: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    warnings = []
    for record in clause_records(sections):
        for risk_type, config in RISKY_PATTERNS.items():
            if not config["pattern"].search(record["text"]):
                continue
            if is_safe_negation(record["text"], config):
                continue
            warnings.append({
                "type": risk_type,
                "severity": config.get("severity", "medium"),
                "section_number": record["section_number"],
                "clause_id": record["clause_id"],
                "message": "Clause matches a general legal-risk pattern and needs human review.",
                "suggested_source_checks": RISKY_SOURCE_HINTS.get(risk_type, []),
                "source_ids": record["source_ids"],
                "text_preview": record["text"][:500],
            })
    return warnings


def validate_constraint_violations(sections: dict[str, dict[str, Any]], rag_debug: dict[str, Any]) -> list[dict[str, Any]]:
    constraints_by_id: dict[str, dict[str, Any]] = {}
    for constraints_obj in constraints_from_rag_debug(rag_debug):
        for constraint in constraints_obj.get("constraints", []):
            cid = constraint.get("id")
            if cid and cid not in constraints_by_id:
                constraints_by_id[cid] = constraint
    warnings = []
    if not constraints_by_id:
        return warnings
    for record in clause_records(sections):
        for cid, constraint in constraints_by_id.items():
            if not constraint_violation(cid, record["text"]):
                continue
            warnings.append({
                "type": "constraint_violation",
                "constraint_id": cid,
                "severity": "high",
                "section_number": record["section_number"],
                "clause_id": record["clause_id"],
                "message": "Clause violates a global or section legal constraint derived from RAG.",
                "prohibited": constraint.get("prohibited"),
                "safe_alternative": constraint.get("safe_alternative"),
                "source_ids": record["source_ids"],
                "constraint_source_ids": constraint.get("source_ids", []),
                "text_preview": record["text"][:500],
            })
    return warnings


def validate_consistency(sections: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    warnings = []
    records = clause_records(sections)
    for negative, positive, warning_type in CONTRADICTION_PAIRS:
        negative_hits = [record for record in records if negative.search(record["text"])]
        positive_hits = [record for record in records if positive.search(record["text"])]
        positive_hits = [record for record in positive_hits if record not in negative_hits and not NEGATION_RE.search(record["text"])]
        if negative_hits and positive_hits:
            warnings.append({
                "type": warning_type,
                "severity": "high",
                "message": "Document contains clauses that appear to conflict across sections.",
                "negative_clause_ids": [record["clause_id"] for record in negative_hits],
                "positive_clause_ids": [record["clause_id"] for record in positive_hits],
            })
    return warnings


def build_allowed_source_ids(rag_debug: dict[str, Any]) -> set[str]:
    allowed = set()
    for results in rag_debug.values():
        if not isinstance(results, list):
            continue
        for item in results:
            if isinstance(item, dict):
                source_id = item.get("chunk_id")
                if source_id:
                    allowed.add(str(source_id))
    return allowed


def validate_legal_warnings(sections: dict[str, dict[str, Any]], rag_debug: dict[str, Any] | None = None) -> dict[str, Any]:
    rag_debug = rag_debug or {}
    warnings = []
    warnings.extend(validate_clause_risks(sections))
    warnings.extend(validate_constraint_violations(sections, rag_debug))
    warnings.extend(validate_consistency(sections))
    if rag_debug:
        warnings.extend(validate_sources(sections, build_allowed_source_ids(rag_debug)))
    return {
        "warning_count": len(warnings),
        "warnings": warnings,
    }


def warning_blockers(legal_warnings: dict[str, Any]) -> list[dict[str, Any]]:
    blockers = []
    for warning in legal_warnings.get("warnings", []):
        if not isinstance(warning, dict):
            continue
        if warning.get("severity") != "high":
            continue
        if warning.get("type") in {
            "full_payment_during_block",
            "monetary_contractual_penalty",
            "unilateral_immediate_termination",
            "silence_acceptance_material_change",
            "constraint_violation",
            "blocking_access_conflict",
            "unknown_source_id",
        }:
            blockers.append(warning)
    return blockers
