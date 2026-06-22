from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


RISK_CATEGORY_PATTERNS: list[tuple[str, str]] = [
    ("consumer_forum_clause", r"sąd|spór|spory|właściw"),
    ("excessive_withdrawal_penalty", r"zaliczk|rezygnac|odstąp|30%|kara\w*\s+umown"),
    ("silence_as_acceptance", r"milczen|brak\s+(sprzeciwu|odpowiedzi)|akceptac|jednostron"),
    ("payment_penalty", r"kara\w*\s+umown[^.]{0,160}(opóźn|zapłat|wynagrodz|pienięż)"),
    ("payment_delay_interest", r"odsetk|wezwan\w*\s+do\s+zapłat|opóźn\w*\s+w\s+zapłat"),
    ("one_sided_termination", r"rozwiąza|wypowied|natychmiast|ważn\w*\s+przyczyn|dowoln\w*\s+pow"),
    ("no_refund_unperformed_service", r"niewykonan|cał\w*\s+wynagrodz|niewykorzystan\w*\s+częś|zwrot"),
]

ACTION_PATTERNS: list[tuple[str, str]] = [
    ("set_exclusive_court_jurisdiction", r"wyłącznie[^.]{0,80}sąd|sąd[^.]{0,80}siedzib"),
    ("set_court_jurisdiction_by_law", r"sąd[^.]{0,120}(powszechnie\s+obowiązującymi\s+przepisami|przepisami\s+prawa)"),
    ("keep_full_deposit", r"zatrzyma\w*\s+cał\w*\s+zaliczk"),
    ("keep_full_deposit_automatically", r"nie\s+zatrzym\w*\s+cał\w*\s+zaliczk\w*\s+automatycz"),
    ("refund_unused_deposit", r"niewykorzystan\w*\s+częś\w*\s+zaliczk\w*[^.]{0,120}(zwrac|zwrot|podlega\s+zwrot)"),
    ("charge_contractual_penalty", r"kar\w*\s+umown"),
    ("unilaterally_change_material_terms", r"jednostron\w*[^.]{0,160}(zmieni|zmian)[^.]{0,160}(termin|miejsce|zakres|cen)"),
    ("treat_silence_as_acceptance", r"(brak\s+(sprzeciwu|odpowiedzi)|milczen)[^.]{0,180}(oznacza|stanowi|traktowan)[^.]{0,80}akcept"),
    ("reject_silence_as_acceptance", r"(brak\s+(sprzeciwu|odpowiedzi)|milczen)[^.]{0,160}nie\s+(oznacza|stanowi|jest\s+traktowan)[^.]{0,80}akcept|nie\s+(oznacza|stanowi)[^.]{0,120}akcept"),
    ("charge_interest", r"odsetk\w*\s+za\s+opóźnienie|odsetek\s+za\s+opóźnienie"),
    ("send_payment_demand", r"wezwan\w*\s+do\s+zapłat"),
    ("terminate_immediately", r"rozwiąza\w*[^.]{0,80}natychmiast|natychmiast[^.]{0,80}rozwiąza"),
    ("terminate_for_important_reason", r"rozwiąza\w*[^.]{0,120}ważn\w*\s+przyczyn|ważn\w*\s+przyczyn[^.]{0,120}rozwiąza"),
    ("keep_full_remuneration", r"zachowa\w*\s+cał\w*\s+wynagrodz|cał\w*\s+wynagrodz[^.]{0,120}niewykonan"),
    ("settle_performed_services_and_refund_unused_payment", r"rozlicz\w*\s+wykonan\w*\s+usług[^.]{0,180}(niewykorzystan|zwrot)|niewykorzystan\w*\s+częś\w*\s+płatnoś\w*\s+podlega\s+zwrot"),
]

ACTION_OBJECTS = {
    "set_exclusive_court_jurisdiction": "court_jurisdiction",
    "set_court_jurisdiction_by_law": "court_jurisdiction",
    "keep_full_deposit": "deposit_or_advance_payment",
    "keep_full_deposit_automatically": "deposit_or_advance_payment",
    "refund_unused_deposit": "deposit_or_advance_payment",
    "charge_contractual_penalty": "contractual_penalty",
    "unilaterally_change_material_terms": "contract_terms",
    "treat_silence_as_acceptance": "acceptance_of_material_change",
    "reject_silence_as_acceptance": "acceptance_of_material_change",
    "charge_interest": "interest_for_delay",
    "send_payment_demand": "payment_demand",
    "terminate_immediately": "contract_termination",
    "terminate_for_important_reason": "contract_termination",
    "keep_full_remuneration": "remuneration",
    "settle_performed_services_and_refund_unused_payment": "remuneration_or_advance_payment",
}

ACTION_MODALITY = {
    "set_exclusive_court_jurisdiction": "permission",
    "set_court_jurisdiction_by_law": "condition",
    "keep_full_deposit": "permission",
    "keep_full_deposit_automatically": "prohibition",
    "refund_unused_deposit": "obligation",
    "charge_contractual_penalty": "permission",
    "unilaterally_change_material_terms": "permission",
    "treat_silence_as_acceptance": "condition",
    "reject_silence_as_acceptance": "prohibition",
    "charge_interest": "right",
    "send_payment_demand": "permission",
    "terminate_immediately": "permission",
    "terminate_for_important_reason": "permission",
    "keep_full_remuneration": "permission",
    "settle_performed_services_and_refund_unused_payment": "obligation",
}

SAFE_ACTION_ALIASES = {
    "reject_silence_as_acceptance": "treat_silence_as_acceptance",
    "keep_full_deposit_automatically": "keep_full_deposit",
}

RISKY_ACTIONS = {
    "set_exclusive_court_jurisdiction",
    "keep_full_deposit",
    "charge_contractual_penalty",
    "unilaterally_change_material_terms",
    "treat_silence_as_acceptance",
    "terminate_immediately",
    "keep_full_remuneration",
}

SAFE_ACTIONS = {
    "set_court_jurisdiction_by_law",
    "refund_unused_deposit",
    "keep_full_deposit_automatically",
    "reject_silence_as_acceptance",
    "charge_interest",
    "send_payment_demand",
    "terminate_for_important_reason",
    "settle_performed_services_and_refund_unused_payment",
}


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def matches(pattern: str, value: str) -> bool:
    return re.search(pattern, value, re.IGNORECASE) is not None


def infer_risk_category(text: str, action: str) -> str:
    for category, pattern in RISK_CATEGORY_PATTERNS:
        if matches(pattern, text):
            return category
    if action in {"set_exclusive_court_jurisdiction", "set_court_jurisdiction_by_law"}:
        return "consumer_forum_clause"
    if action in {"charge_interest", "send_payment_demand"}:
        return "payment_delay_interest"
    if action == "charge_contractual_penalty":
        return "payment_penalty" if matches(r"opóźn|zapłat|wynagrodz|pienięż", text) else "contractual_penalty"
    return "unknown"


def infer_obligation_kind(text: str, action: str) -> str:
    if matches(r"zapłat|wynagrodz|pienięż|odsetk|zaliczk|płatnoś", text):
        return "monetary"
    if action in {"set_exclusive_court_jurisdiction", "set_court_jurisdiction_by_law", "unilaterally_change_material_terms", "treat_silence_as_acceptance", "reject_silence_as_acceptance", "terminate_immediately", "terminate_for_important_reason"}:
        return "non_monetary"
    return "unknown"


def infer_trigger(text: str, action: str) -> str | None:
    trigger_patterns = [
        ("late_monetary_payment", r"opóźn\w*\s+w\s+zapłat|opóźn\w*[^.]{0,80}(wynagrodz|pienięż|zapłat)"),
        ("customer_withdrawal", r"rezygnac|odstąp"),
        ("no_objection_within_3_days", r"brak\s+sprzeciwu[^.]{0,80}3\s*dni"),
        ("no_response_or_silence", r"brak\s+odpowiedzi|milczen"),
        ("material_change", r"istotn\w*\s+zmian|termin|miejsce|zakres|cen"),
        ("any_reason", r"dowoln\w*\s+pow"),
        ("important_reason", r"ważn\w*\s+przyczyn"),
        ("contract_termination", r"rozwiąza|wypowied"),
        ("consumer_dispute", r"konsument[^.]{0,120}(spór|sąd)|spór[^.]{0,120}konsument"),
        ("contract_dispute", r"spór|spory|sąd"),
    ]
    for trigger, pattern in trigger_patterns:
        if matches(pattern, text):
            return trigger
    if action == "charge_contractual_penalty" and infer_obligation_kind(text, action) == "monetary":
        return "late_monetary_payment"
    return None


def infer_condition(text: str, action: str) -> str | None:
    if matches(r"siedzib", text) and "court" in ACTION_OBJECTS.get(action, ""):
        return "entrepreneur_seat_court"
    if matches(r"powszechnie\s+obowiązującymi\s+przepisami|przepisami\s+prawa", text):
        return "general_applicable_law"
    if matches(r"mał\w*\s+częś|część\s+prac", text):
        return "small_part_of_work_performed"
    if matches(r"niezależnie\s+od\s+odset", text):
        return "independent_of_interest"
    if matches(r"bez\s+analogicznego\s+praw", text):
        return "no_analogous_customer_right"
    if matches(r"niewykonan\w*\s+jeszcze\s+usług", text):
        return "unperformed_services_remain"
    if matches(r"wykonan\w*\s+usług|rzeczywist\w*\s+koszt|uzasadnion\w*\s+koszt", text):
        return "performed_services_and_justified_costs_only"
    return None


def infer_amount(text: str) -> str | None:
    percentage = re.search(r"\b\d+\s*%", text)
    if percentage:
        return percentage.group(0).replace(" ", "")
    if matches(r"cał\w*\s+wynagrodz", text):
        return "full remuneration"
    if matches(r"cał\w*\s+zaliczk", text):
        return "full deposit"
    return None


def canonical_action(action: str) -> str:
    return SAFE_ACTION_ALIASES.get(action, action)


def expected_safety_from_action(action: str, modality: str) -> str:
    if action in SAFE_ACTIONS or modality == "prohibition":
        return "safe"
    if action in RISKY_ACTIONS:
        return "risky"
    return "neutral"


def object_from_action(text: str, action: str, index: int, *, origin: str = "draft", source_ids: list[str] | None = None) -> dict[str, Any]:
    normalized = normalize_text(text)
    modality = ACTION_MODALITY.get(action, "condition")
    risk_category = infer_risk_category(normalized, action)
    obligation_kind = infer_obligation_kind(normalized, action)
    return {
        "object_id": f"{origin}_obj_{index}",
        "origin": origin,
        "text_span": text.strip(),
        "source_ids": source_ids or [],
        "jurisdiction": "PL",
        "language": "pl",
        "actor": None,
        "counterparty": None,
        "modality": modality,
        "action": action,
        "canonical_action": canonical_action(action),
        "legal_object": ACTION_OBJECTS.get(action, "unknown"),
        "trigger": infer_trigger(normalized, action),
        "condition": infer_condition(normalized, action),
        "exception": None,
        "temporal_scope": None,
        "amount": infer_amount(normalized),
        "payment_kind": "monetary_payment" if obligation_kind == "monetary" else None,
        "obligation_kind": obligation_kind,
        "risk_category": risk_category,
        "expected_safety": expected_safety_from_action(action, modality),
        "confidence": 0.74,
        "extraction_method": "deterministic_pattern",
    }


def extract_legal_objects(text: str, *, origin: str = "draft", source_ids: list[str] | None = None) -> list[dict[str, Any]]:
    normalized = normalize_text(text)
    objects: list[dict[str, Any]] = []
    for action, pattern in ACTION_PATTERNS:
        if matches(pattern, normalized):
            objects.append(object_from_action(text, action, len(objects) + 1, origin=origin, source_ids=source_ids))
    if objects:
        return objects
    risk_category = infer_risk_category(normalized, "unknown")
    if risk_category != "unknown":
        return [
            {
                "object_id": f"{origin}_obj_1",
                "origin": origin,
                "text_span": text.strip(),
                "source_ids": source_ids or [],
                "jurisdiction": "PL",
                "language": "pl",
                "actor": None,
                "counterparty": None,
                "modality": "condition",
                "action": "unknown",
                "canonical_action": "unknown",
                "legal_object": "unknown",
                "trigger": infer_trigger(normalized, "unknown"),
                "condition": None,
                "exception": None,
                "temporal_scope": None,
                "amount": infer_amount(normalized),
                "payment_kind": None,
                "obligation_kind": infer_obligation_kind(normalized, "unknown"),
                "risk_category": risk_category,
                "expected_safety": "neutral",
                "confidence": 0.35,
                "extraction_method": "deterministic_risk_category_only",
            }
        ]
    return []


def extract_legal_objects_from_label(label: dict[str, Any]) -> list[dict[str, Any]]:
    return extract_legal_objects(str(label.get("input_text") or label.get("claim") or ""), origin="label")


def normalize_expected_object(value: dict[str, Any], index: int) -> dict[str, Any]:
    normalized = dict(value)
    normalized.setdefault("object_id", f"expected_obj_{index}")
    action = str(normalized.get("action") or "unknown")
    normalized.setdefault("canonical_action", canonical_action(action))
    normalized.setdefault("legal_object", ACTION_OBJECTS.get(action, normalized.get("legal_object") or "unknown"))
    normalized.setdefault("expected_safety", "safe" if normalized.get("expected_verdict") == "supported" else "risky")
    return normalized


@dataclass(frozen=True)
class ObjectMatch:
    expected: dict[str, Any]
    extracted: dict[str, Any] | None
    score: float
    reasons: list[str]


def object_match_score(expected: dict[str, Any], extracted: dict[str, Any]) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []
    fields = [
        ("canonical_action", 0.42),
        ("legal_object", 0.18),
        ("risk_category", 0.16),
        ("obligation_kind", 0.10),
        ("trigger", 0.08),
        ("modality", 0.06),
    ]
    for field, weight in fields:
        expected_value = expected.get(field)
        extracted_value = extracted.get(field)
        if expected_value and extracted_value and expected_value == extracted_value:
            score += weight
            reasons.append(f"{field}_match")
    return round(score, 4), reasons


def match_expected_objects(expected_objects: list[dict[str, Any]], extracted_objects: list[dict[str, Any]]) -> list[ObjectMatch]:
    matches: list[ObjectMatch] = []
    used: set[int] = set()
    for index, raw_expected in enumerate(expected_objects, start=1):
        expected = normalize_expected_object(raw_expected, index)
        best_index = None
        best_score = -1.0
        best_reasons: list[str] = []
        for candidate_index, extracted in enumerate(extracted_objects):
            if candidate_index in used:
                continue
            score, reasons = object_match_score(expected, extracted)
            if score > best_score:
                best_index = candidate_index
                best_score = score
                best_reasons = reasons
        if best_index is not None and best_score >= 0.42:
            used.add(best_index)
            matches.append(ObjectMatch(expected, extracted_objects[best_index], best_score, best_reasons))
        else:
            matches.append(ObjectMatch(expected, None, max(best_score, 0.0), best_reasons))
    return matches
