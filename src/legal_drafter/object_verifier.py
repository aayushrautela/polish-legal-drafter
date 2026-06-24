from __future__ import annotations

from typing import Any

from .legal_objects import match_expected_objects
from .source_semantics import NEGATIVE_POLARITIES, POSITIVE_POLARITIES


RISKY_VERDICTS = {"risky", "contradicted"}
SAFE_VERDICTS = {"supported"}


def expected_final_verdict(value: str) -> str:
    if value in {"supported", "supported_or_safe"}:
        return "supported"
    if value in {"risky", "contradicted", "contradicted_or_risky"}:
        return "risky"
    return "insufficient"


def evidence_by_id(evidence: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(item.get("source_id")): item for item in evidence if item.get("source_id")}


def object_evidence(objects: list[dict[str, Any]], evidence: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_id = evidence_by_id(evidence)
    out: dict[str, list[dict[str, Any]]] = {}
    for obj in objects:
        expected_ids = [str(item) for item in obj.get("expected_evidence_ids") or []]
        out[str(obj.get("object_id"))] = [by_id[source_id] for source_id in expected_ids if source_id in by_id]
    return out


def infer_verdict_from_object(obj: dict[str, Any], evidence_items: list[dict[str, Any]]) -> tuple[str, str, float]:
    expected_safety = str(obj.get("expected_safety") or "neutral")
    action = str(obj.get("action") or "")
    matched_action = str(obj.get("matched_action") or action)
    obligation_kind = str(obj.get("obligation_kind") or "")
    risk_category = str(obj.get("risk_category") or "")
    polarities = {str(item.get("polarity") or "unknown") for item in evidence_items}

    if action == "charge_contractual_penalty" and obligation_kind == "monetary":
        return "risky", "contractual_penalty_attached_to_monetary_obligation", 0.95
    if matched_action in {"reject_silence_as_acceptance", "keep_full_deposit_automatically"} and polarities & NEGATIVE_POLARITIES:
        return "supported", "protective_object_contrasts_with_negative_example_source", 0.86
    if action in {"set_exclusive_court_jurisdiction", "treat_silence_as_acceptance", "keep_full_deposit", "unilaterally_change_material_terms"} and polarities & NEGATIVE_POLARITIES:
        return "risky", "draft_object_matches_negative_example_source", 0.9
    if action in {"terminate_immediately", "keep_full_remuneration"} and risk_category in {"one_sided_termination", "no_refund_unperformed_service"}:
        return "risky", "one_sided_or_no_refund_object_requires_repair", 0.82
    if expected_safety == "safe":
        if polarities & NEGATIVE_POLARITIES:
            return "supported", "protective_object_contrasts_with_negative_example_source", 0.82
        if polarities & POSITIVE_POLARITIES:
            return "supported", "object_supported_by_normative_source", 0.82
        return "supported", "safe_object_without_contrary_evidence", 0.64
    if expected_safety == "risky":
        if polarities & NEGATIVE_POLARITIES:
            return "risky", "risky_object_matches_negative_example_source", 0.82
        if polarities & POSITIVE_POLARITIES:
            return "risky", "risky_object_conflicts_with_normative_source_or_conditions", 0.72
        return "needs_review", "risky_object_without_strong_evidence", 0.5
    return "insufficient", "object_not_classified", 0.35


def verify_expected_objects(
    expected_objects: list[dict[str, Any]],
    extracted_objects: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    matches = match_expected_objects(expected_objects, extracted_objects)
    by_expected = object_evidence([match.expected for match in matches], evidence)
    results: list[dict[str, Any]] = []
    for match in matches:
        if match.extracted is None:
            expected_verdict = expected_final_verdict(str(match.expected.get("expected_verdict") or "insufficient"))
            results.append(
                {
                    "object_id": match.expected.get("object_id"),
                    "expected_verdict": expected_verdict,
                    "predicted_verdict": "insufficient",
                    "pass": expected_verdict == "insufficient",
                    "match_score": match.score,
                    "match_reasons": match.reasons,
                    "reason": "expected_object_not_extracted",
                    "expected_object": match.expected,
                    "extracted_object": None,
                    "evidence_ids": [],
                    "confidence": 0.0,
                }
            )
            continue
        evidence_items = by_expected.get(str(match.expected.get("object_id")), [])
        predicted_verdict, reason, confidence = infer_verdict_from_object(
            {**match.expected, "matched_action": match.extracted.get("action")},
            evidence_items,
        )
        expected_verdict = expected_final_verdict(str(match.expected.get("expected_verdict") or "insufficient"))
        results.append(
            {
                "object_id": match.expected.get("object_id"),
                "expected_verdict": expected_verdict,
                "predicted_verdict": predicted_verdict,
                "pass": predicted_verdict == expected_verdict,
                "match_score": match.score,
                "match_reasons": match.reasons,
                "reason": reason,
                "expected_object": match.expected,
                "extracted_object": match.extracted,
                "evidence_ids": [item.get("source_id") for item in evidence_items],
                "evidence_polarities": [item.get("polarity") for item in evidence_items],
                "confidence": confidence,
            }
        )
    return results


def summarize_object_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    passed = sum(1 for item in results if item.get("pass"))
    return {
        "objects": total,
        "correct": passed,
        "accuracy": round(passed / total, 4) if total else 0.0,
    }
