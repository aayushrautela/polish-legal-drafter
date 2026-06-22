from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

PLACEHOLDER_RE = re.compile(r"\[\[[A-Z0-9_]+\]\]")
BAD_PLACEHOLDER_RE = re.compile(r"\[\[[^\]]+\]\]")  # any double-bracket placeholder (then we validate allowed chars)
FORBIDDEN_SUBSTRINGS = ["[[WYPELNIJ:", "```"]
FORBIDDEN_SIGNATURE_TOKENS = ["Data:", "Miejsce:", "Podpis", "Podpisy"]

class ValidationError(ValueError):
    def __init__(self, errors: List[str]):
        super().__init__("; ".join(errors))
        self.errors = errors

def find_placeholders(text: str) -> List[str]:
    return re.findall(r"\[\[([^\]]+)\]\]", text)

def validate_placeholder_format(text: str) -> List[str]:
    errors: List[str] = []
    if any(bad in text for bad in FORBIDDEN_SUBSTRINGS):
        for bad in FORBIDDEN_SUBSTRINGS:
            if bad in text:
                errors.append(f"Forbidden substring present: {bad}")
    # Identify placeholders and ensure they match the strict regex
    for raw in re.findall(r"\[\[[^\]]+\]\]", text):
        if not PLACEHOLDER_RE.fullmatch(raw):
            errors.append(f"Invalid placeholder token: {raw} (use only [[A-Z0-9_]])")
    return errors

def validate_placeholders_allowlist(text: str, allowlist: Set[str]) -> List[str]:
    errors: List[str] = []
    for token in find_placeholders(text):
        if token not in allowlist:
            errors.append(f"Placeholder not in allowlist: [[{token}]]")
    return errors

def validate_no_signature_tokens(text: str) -> List[str]:
    errors: List[str] = []
    for tok in FORBIDDEN_SIGNATURE_TOKENS:
        # We forbid the colon variants strongly; plain 'data' word is allowed.
        if tok in text:
            errors.append(f"Forbidden signature token in non-signature section: {tok}")
    return errors


def normalise_for_matching(value: str) -> str:
    value = unicodedata.normalize("NFKD", value.lower())
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", value)


def keyword_matches(text: str, keyword: str) -> bool:
    normal_text = normalise_for_matching(text)
    normal_keyword = normalise_for_matching(keyword)
    if normal_keyword in normal_text:
        return True
    stem = normal_keyword
    for suffix in ("ość", "osc", "enie", "anie", "acja", "acje", "acje", "ego", "ych", "ami", "ach", "owa", "owe", "owy", "nie", "ia", "ie", "ci", "sc", "ść"):
        if len(stem) > len(suffix) + 3 and stem.endswith(normalise_for_matching(suffix)):
            stem = stem[: -len(normalise_for_matching(suffix))]
            break
    return len(stem) >= 4 and stem in normal_text


def validation_min_words(min_words_total: int) -> int:
    return max(1, int(min_words_total * 0.65))

def validate_section_json(
    section_obj: Dict[str, Any],
    section_number: int,
    min_clauses: int,
    min_words_total: int,
    must_include_any: List[str],
    placeholder_allowlist: Set[str],
    strict_allowlist: bool = True,
    allowed_source_ids: Optional[Set[str]] = None,
) -> None:
    errors: List[str] = []

    if not isinstance(section_obj, dict):
        raise ValidationError(["Section JSON is not an object"])

    if section_obj.get("section_number") != section_number:
        errors.append(f"section_number mismatch (expected {section_number})")

    clauses = section_obj.get("clauses")
    if not isinstance(clauses, list) or not clauses:
        errors.append("clauses must be a non-empty list")

    texts: List[str] = []
    if isinstance(clauses, list):
        for idx, c in enumerate(clauses):
            if not isinstance(c, dict):
                errors.append(f"clause[{idx}] is not an object")
                continue
            cid = c.get("id")
            txt = c.get("text")
            if not isinstance(cid, str) or not cid.startswith(f"{section_number}."):
                errors.append(f"Invalid clause id: {cid} (must start with '{section_number}.')")
            if not isinstance(txt, str) or len(txt.strip()) < 5:
                errors.append(f"Invalid clause text for id={cid}")
            else:
                texts.append(txt)
            source_ids = c.get("source_ids", [])
            if source_ids is not None:
                if not isinstance(source_ids, list) or any(not isinstance(item, str) for item in source_ids):
                    errors.append(f"Invalid source_ids for id={cid} (must be a list of strings)")
                elif allowed_source_ids is not None:
                    for source_id in source_ids:
                        if source_id not in allowed_source_ids:
                            errors.append(f"Unknown source_id for id={cid}: {source_id}")

    if isinstance(clauses, list) and len(clauses) < min_clauses:
        errors.append(f"Too few clauses: {len(clauses)} < {min_clauses}")

    full_text = "\n".join(texts)

    # Placeholder checks
    errors.extend(validate_placeholder_format(full_text))
    errors.extend(validate_no_signature_tokens(full_text))
    if strict_allowlist:
        errors.extend(validate_placeholders_allowlist(full_text, placeholder_allowlist))

    # Length checks (very rough but effective)
    words = [w for w in re.split(r"\s+", full_text.strip()) if w]
    min_words_required = validation_min_words(min_words_total)
    if len(words) < min_words_required:
        errors.append(f"Section too short: {len(words)} words < {min_words_required} required ({min_words_total} target)")

    # Topic coverage heuristic
    if must_include_any:
        hits = sum(1 for k in must_include_any if keyword_matches(full_text, k))
        required_hits = max(1, min(2, len(must_include_any) // 3))
        if hits < required_hits:
            errors.append(
                "Section may be off-topic: too few keyword hits. " +
                f"hits={hits}, required={required_hits}, keywords={must_include_any}"
            )

    if errors:
        raise ValidationError(errors)

def validate_translation_alignment(pl_section: Dict[str, Any], en_section: Dict[str, Any], placeholder_allowlist: Set[str], strict_allowlist: bool = True) -> None:
    errors: List[str] = []
    if pl_section.get("section_number") != en_section.get("section_number"):
        errors.append("section_number mismatch between PL and EN")

    pl_clauses = pl_section.get("clauses", [])
    en_clauses = en_section.get("clauses", [])
    pl_ids = [c.get("id") for c in pl_clauses if isinstance(c, dict)]
    en_ids = [c.get("id") for c in en_clauses if isinstance(c, dict)]
    if pl_ids != en_ids:
        errors.append(f"Clause IDs differ (PL vs EN):\nPL={pl_ids}\nEN={en_ids}")

    # Placeholders must be preserved (same set)
    pl_text = "\n".join([c.get("text", "") for c in pl_clauses if isinstance(c, dict)])
    en_text = "\n".join([c.get("text", "") for c in en_clauses if isinstance(c, dict)])

    pl_ph = set(find_placeholders(pl_text))
    en_ph = set(find_placeholders(en_text))
    if pl_ph != en_ph:
        errors.append(f"Placeholder set mismatch PL vs EN:\nPL={sorted(pl_ph)}\nEN={sorted(en_ph)}")

    # Ensure EN doesn't introduce invalid placeholders
    errors.extend(validate_placeholder_format(en_text))
    if strict_allowlist:
        errors.extend(validate_placeholders_allowlist(en_text, placeholder_allowlist))

    if errors:
        raise ValidationError(errors)
