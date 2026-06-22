from __future__ import annotations

import re
from typing import Any

LEGAL_RISK_RE = re.compile(
    r"\b(kara\w*\s+umown|odsetk|opóźn|opozn|zapłat|zaplac|wynagrodzen|zaliczk|kaucj|zwrot|rezygnac|odstąp|odstap|wypowied|rozwiąz|rozwiaz|natychmiast|sąd|sad|właściw|wlasciw|konsument|jednostron|milczen|brak\s+(sprzeciwu|odpowiedzi)|akceptac|odpowiedzialno|niedozwolon|nieważn|niewazn)\b",
    re.IGNORECASE,
)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-ZĄĆĘŁŃÓŚŹŻ0-9])")


def split_claims(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = [part.strip(" ;") for part in SENTENCE_SPLIT_RE.split(text) if part.strip(" ;")]
    claims: list[str] = []
    for part in parts:
        if len(part) > 420:
            subparts = [item.strip(" ;") for item in re.split(r";\s+", part) if item.strip(" ;")]
            claims.extend(subparts or [part])
        else:
            claims.append(part)
    return claims


def needs_verification(text: str) -> bool:
    return bool(LEGAL_RISK_RE.search(text))


def extract_claims_from_clause(
    clause: dict[str, Any],
    *,
    case_id: str = "",
    section_id: str = "",
    section_title: str = "",
) -> list[dict[str, Any]]:
    clause_id = str(clause.get("id") or "")
    text = str(clause.get("text") or "")
    source_ids = clause.get("source_ids") if isinstance(clause.get("source_ids"), list) else []
    claims = []
    for index, claim_text in enumerate(split_claims(text), start=1):
        verify = needs_verification(claim_text)
        claims.append({
            "claim_id": "::".join(part for part in [case_id, section_id, clause_id, f"claim{index}"] if part),
            "case_id": case_id,
            "section_id": section_id,
            "section_title": section_title,
            "clause_id": clause_id,
            "claim_index": index,
            "claim_text": claim_text,
            "claim_type": "legal_risk" if verify else "administrative",
            "source_ids_from_clause": [str(value) for value in source_ids if isinstance(value, str)],
            "needs_verification": verify,
        })
    return claims


def extract_claims_from_sections(sections: dict[str, dict[str, Any]], *, case_id: str = "") -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    for section_id, section in sections.items():
        section_title = str(section.get("title") or "")
        for clause in section.get("clauses", []):
            if isinstance(clause, dict):
                claims.extend(extract_claims_from_clause(clause, case_id=case_id, section_id=str(section_id), section_title=section_title))
    return claims


def extract_claim_from_label(label: dict[str, Any]) -> dict[str, Any]:
    claim = {
        "claim_id": str(label.get("label_id") or ""),
        "case_id": str(label.get("case_id") or ""),
        "section_id": "",
        "section_title": "",
        "clause_id": "",
        "claim_index": 1,
        "claim_text": str(label.get("claim") or ""),
        "claim_type": "legal_risk",
        "source_ids_from_clause": [],
        "needs_verification": True,
    }
    return claim
