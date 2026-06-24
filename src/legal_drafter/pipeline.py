from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .doc_types import DocType
from .io_utils import ensure_dir, read_json, read_text, write_json, write_text
from .json_utils import JsonParseError, dumps_pretty, loads_best_effort
from .document_specs import requirements_summary
from .text_template import render_template
from .validators import ValidationError, validate_section_json, validate_translation_alignment
from .renderer import load_locale, render_markdown
from .rag import build_rag_index, format_rag_context, result_summary, search_section_context
from .legal_constraints import derive_legal_constraints, format_legal_constraints, merge_legal_constraints
from .legal_warnings import build_allowed_source_ids, validate_legal_warnings, warning_blockers

@dataclass
class PromptBundle:
    plan_prompt: str
    section_prompt: str
    translate_prompt: str
    repair_section_prompt: str

def load_prompts(prompts_dir: str | Path) -> PromptBundle:
    p = Path(prompts_dir)
    return PromptBundle(
        plan_prompt=(p / "plan_prompt.txt").read_text(encoding="utf-8"),
        section_prompt=(p / "section_prompt.txt").read_text(encoding="utf-8"),
        translate_prompt=(p / "translate_prompt.txt").read_text(encoding="utf-8"),
        repair_section_prompt=(p / "repair_section_prompt.txt").read_text(encoding="utf-8"),
    )

def _pretty_allowlist(allowlist: List[str]) -> str:
    return ", ".join([f"[[{x}]]" for x in allowlist])


def _close_truncated_section_json(raw: str, section_number: int, section_title: str) -> str:
    if not raw.strip().startswith("{"):
        return raw
    if raw.count("{") <= raw.count("}"):
        return raw
    trimmed = raw.rstrip()
    last_clause_end = trimmed.rfind("}\n")
    if last_clause_end == -1:
        last_clause_end = trimmed.rfind("}")
    if last_clause_end == -1:
        return json.dumps({
            "section_number": section_number,
            "section_title": section_title,
            "clauses": [],
        }, ensure_ascii=False)
    body = trimmed[: last_clause_end + 1].rstrip().rstrip(",")
    if '"clauses"' not in body:
        return raw
    return body + "\n  ]\n}"


def _normalise_clause_source_ids(section_obj: Dict[str, Any]) -> None:
    for clause in section_obj.get("clauses", []):
        if isinstance(clause, dict) and "source_ids" not in clause:
            clause["source_ids"] = []


def _blocker_mentions_section(blocker: Dict[str, Any], section_key: str) -> bool:
    if str(blocker.get("section_number", "")) == section_key:
        return True
    prefix = f"{section_key}."
    for key in ("clause_id", "negative_clause_ids", "positive_clause_ids"):
        value = blocker.get(key)
        if isinstance(value, str) and value.startswith(prefix):
            return True
        if isinstance(value, list) and any(str(item).startswith(prefix) for item in value):
            return True
    return False


def _section_blockers(sections: Dict[str, Dict[str, Any]], rag_debug: Dict[str, Any], section_key: str) -> List[Dict[str, Any]]:
    legal_warnings = validate_legal_warnings(sections, rag_debug)
    return [blocker for blocker in warning_blockers(legal_warnings) if _blocker_mentions_section(blocker, section_key)]

def run_phase1_contract_pipeline(
    *,
    engine,  # Modal remote object with .generate.remote(prompt, **gen_cfg)
    doc: DocType,
    facts: str,
    requirements: Dict[str, Any],
    placeholders: Dict[str, Any],
    prompts: PromptBundle,
    locales_pl_path: str | Path,
    locales_en_path: str | Path,
    out_dir: str | Path,
    bilingual: bool = True,
    strict_placeholder_allowlist: bool = True,
    rag_paths: Optional[List[str | Path]] = None,
    rag_top_k: int = 6,
) -> Dict[str, Any]:
    out = ensure_dir(out_dir)

    allowlist = placeholders.get("allowlist", [])
    allowset: Set[str] = set(allowlist)

    # --- Load locales (headings + deterministic blocks) ---
    loc_pl = load_locale(locales_pl_path).get(doc.id)
    loc_en = load_locale(locales_en_path).get(doc.id) if bilingual else None
    if not loc_pl:
        raise ValueError(f"Missing locale for doc_type_id={doc.id} in {locales_pl_path}")
    if bilingual and not loc_en:
        raise ValueError(f"Missing locale for doc_type_id={doc.id} in {locales_en_path}")

    template_headings = " | ".join([f"## {i}. {t}" for i, t in enumerate(doc.sections, start=1)])

    # --- Plan step (optional but helpful for debugging) ---
    req_summary = requirements_summary(requirements)
    rag_index = build_rag_index(rag_paths) if rag_paths else None
    rag_debug: Dict[str, Any] = {}
    global_constraints_obj: Dict[str, Any] = {"constraints": []}
    global_constraints_context = "Brak globalnych ograniczeń, bo RAG nie jest włączony."
    if rag_index:
        global_rag_results = search_section_context(
            rag_index,
            doc_type=doc.id,
            doc_title=doc.title,
            section_number="6",
            section_title="Globalne ryzyka prawne z faktów użytkownika",
            facts=facts,
            section_requirements={"topics": ["klauzule niedozwolone", "kara umowna", "zmiana umowy", "wypowiedzenie", "sąd", "rozliczenie"]},
            top_k=max(rag_top_k, 10),
        )
        constraint_inputs = [
            derive_legal_constraints(
                facts=facts,
                section_number=section_number,
                section_title=title,
                rag_results=global_rag_results,
                rag_index=rag_index,
            )
            for section_number, title in (
                ("5", "Płatność, kary i rozliczenie"),
                ("6", "Zmiana, wypowiedzenie i rozwiązanie"),
                ("7", "Postanowienia końcowe, sąd i konsument"),
            )
        ]
        global_constraints_obj = merge_legal_constraints(*constraint_inputs)
        global_constraints_context = format_legal_constraints(global_constraints_obj).replace(
            "WYKRYTE OGRANICZENIA PRAWNE DLA TEJ SEKCJI",
            "GLOBALNE OGRANICZENIA PRAWNE DLA CAŁEGO DOKUMENTU",
        )
        if global_constraints_obj.get("constraints"):
            rag_debug["global_constraints"] = global_constraints_obj

    plan_vars = {
        "DOC_TYPE_ID": doc.id,
        "DOC_TITLE": doc.title,
        "TEMPLATE_HEADINGS": template_headings,
        "FACTS": facts,
        "PLACEHOLDER_ALLOWLIST": _pretty_allowlist(allowlist),
        "REQUIREMENTS_SUMMARY": req_summary,
    }
    plan_prompt = render_template(prompts.plan_prompt, plan_vars)
    plan_raw = engine.generate.remote(
        plan_prompt,
        max_new_tokens=650,
        temperature=0.35,
        top_p=0.85,
        top_k=20,
        enable_thinking=False,
    )
    try:
        plan_obj = loads_best_effort(plan_raw)
    except JsonParseError as e:
        plan_obj = {"error": str(e), "raw": plan_raw}

    write_json(out / "plan.json", plan_obj)

    # --- Section generation (Polish) ---
    sections_pl: Dict[str, Dict[str, Any]] = {}
    failures: Dict[str, Any] = {}

    gen_cfg = requirements.get("generation", {})
    max_new_tokens = int(gen_cfg.get("per_section_max_new_tokens", 1100))
    max_retries = int(gen_cfg.get("max_retries_per_section", 2))

    for idx, sec_title in enumerate(doc.sections, start=1):
        sec_key = str(idx)
        sec_req = requirements.get("sections", {}).get(sec_key, {})

        # Deterministic sections
        if sec_req.get("mode") in ("deterministic_party_block", "deterministic_signature_block"):
            continue

        min_clauses = int(sec_req.get("min_clauses", 4))
        min_words_total = int(sec_req.get("min_words_total", 160))
        must_include_any = list(sec_req.get("must_include_any", []))

        rag_context = "RAG nie jest włączony dla tego uruchomienia."
        legal_constraints_context = "Brak warstwy constraints, bo RAG nie jest włączony."
        allowed_source_ids: Optional[Set[str]] = None
        if rag_index:
            rag_results = search_section_context(
                rag_index,
                doc_type=doc.id,
                doc_title=doc.title,
                section_number=sec_key,
                section_title=sec_title,
                facts=facts,
                section_requirements=sec_req,
                top_k=rag_top_k,
            )
            constraints_obj = derive_legal_constraints(
                facts=facts,
                section_number=sec_key,
                section_title=sec_title,
                rag_results=rag_results,
                rag_index=rag_index,
            )
            constraints_obj = merge_legal_constraints(global_constraints_obj, constraints_obj)
            rag_context = format_rag_context(rag_results)
            legal_constraints_context = format_legal_constraints(constraints_obj)
            rag_debug[sec_key] = result_summary(rag_results)
            if constraints_obj.get("constraints"):
                rag_debug[sec_key + "_constraints"] = constraints_obj
            allowed_source_ids = build_allowed_source_ids({sec_key: rag_debug[sec_key]})

        section_vars = {
            "DOC_TYPE_ID": doc.id,
            "DOC_TITLE": doc.title,
            "SECTION_NUMBER": str(idx),
            "SECTION_TITLE": sec_title,
            "FACTS": facts,
            "RAG_CONTEXT": rag_context,
            "GLOBAL_LEGAL_CONSTRAINTS": global_constraints_context,
            "LEGAL_CONSTRAINTS": legal_constraints_context,
            "PLACEHOLDER_ALLOWLIST": _pretty_allowlist(allowlist),
            "SECTION_REQUIREMENTS_JSON": json.dumps(sec_req, ensure_ascii=False, indent=2),
            "PLAN_JSON": json.dumps(plan_obj, ensure_ascii=False, indent=2) if isinstance(plan_obj, dict) else str(plan_obj),
            "MIN_CLAUSES": str(min_clauses),
            "MIN_WORDS_TOTAL": str(min_words_total),
        }

        def try_generate(prompt_text: str) -> Tuple[Optional[Dict[str, Any]], str]:
            raw = engine.generate.remote(
                prompt_text,
                max_new_tokens=max_new_tokens,
                temperature=0.35,
                top_p=0.85,
                top_k=20,
                enable_thinking=False,
            )
            try:
                obj = loads_best_effort(raw)
                if not isinstance(obj, dict):
                    return None, raw
                _normalise_clause_source_ids(obj)
                return obj, raw
            except JsonParseError:
                return None, raw

        prompt_text = render_template(prompts.section_prompt, section_vars)
        attempt = 0
        last_raw = ""
        last_errors: List[str] = []
        sec_obj: Optional[Dict[str, Any]] = None

        while attempt <= max_retries:
            attempt += 1
            candidate, raw = try_generate(prompt_text)
            last_raw = raw

            if candidate is None:
                last_errors = ["Model output was not valid JSON"]
            else:
                try:
                    validate_section_json(
                        candidate,
                        section_number=idx,
                        min_clauses=min_clauses,
                        min_words_total=min_words_total,
                        must_include_any=must_include_any,
                        placeholder_allowlist=allowset,
                        strict_allowlist=strict_placeholder_allowlist,
                        allowed_source_ids=allowed_source_ids,
                    )
                    candidate_sections = dict(sections_pl)
                    candidate_sections[sec_key] = candidate
                    blockers = _section_blockers(candidate_sections, rag_debug, sec_key)
                    if blockers:
                        last_errors = [
                            "Legal warning blocker must be repaired: "
                            + json.dumps(blocker, ensure_ascii=False)
                            for blocker in blockers
                        ]
                    else:
                        sec_obj = candidate
                        break
                except ValidationError as ve:
                    last_errors = ve.errors

            if attempt <= max_retries:
                repair_vars = dict(section_vars)
                if candidate is None:
                    last_raw = _close_truncated_section_json(last_raw, idx, sec_title)
                repair_vars.update({
                    "ERRORS": json.dumps(last_errors, ensure_ascii=False, indent=2),
                    "PREVIOUS_OUTPUT": last_raw,
                })
                prompt_text = render_template(prompts.repair_section_prompt, repair_vars)

        if sec_obj is None:
            failures[sec_key] = {
                "section_title": sec_title,
                "errors": last_errors,
                "raw": last_raw[:5000],
            }
        else:
            sections_pl[sec_key] = sec_obj

    write_json(out / "sections_pl.json", sections_pl)
    if rag_debug:
        write_json(out / "rag_sources.json", rag_debug)
    legal_warnings = validate_legal_warnings(sections_pl, rag_debug)
    warning_failures = warning_blockers(legal_warnings)
    write_json(out / "legal_warnings.json", legal_warnings)
    if warning_failures:
        write_json(out / "legal_warning_blockers.json", warning_failures)
    if failures:
        write_json(out / "failures_pl.json", failures)

    # Render Polish markdown deterministically
    draft_pl = render_markdown(
        doc_type_id=doc.id,
        doc_title=loc_pl["title"],
        section_titles=loc_pl["sections"],
        sections=sections_pl,
        party_lines=loc_pl["party_lines"],
        signature_lines=loc_pl["signature_lines"],
        safe_sentences=loc_pl.get("final_safe_sentences", []),
        append_safe_to_section_7=bool(requirements.get("sections", {}).get("7", {}).get("append_safe_sentences", True)),
    )
    write_text(out / "draft_pl.md", draft_pl)

    # --- Translation step (English) ---
    sections_en: Dict[str, Dict[str, Any]] = {}
    failures_en: Dict[str, Any] = {}

    if bilingual:
        translate_max_new_tokens = int(gen_cfg.get("translate_max_new_tokens", 900))

        for sec_key, pl_obj in sections_pl.items():
            idx = int(sec_key)
            sec_req = requirements.get("sections", {}).get(sec_key, {})
            translate_vars = {
                "PLACEHOLDER_ALLOWLIST": _pretty_allowlist(allowlist),
                "SECTION_JSON_PL": json.dumps(pl_obj, ensure_ascii=False, indent=2),
            }
            tprompt = render_template(prompts.translate_prompt, translate_vars)
            raw_en = engine.generate.remote(
                tprompt,
                max_new_tokens=translate_max_new_tokens,
                temperature=0.30,
                top_p=0.85,
                top_k=20,
                enable_thinking=False,
            )
            try:
                en_obj = loads_best_effort(raw_en)
                if not isinstance(en_obj, dict):
                    raise JsonParseError("EN output not an object")
                # Validate alignment
                try:
                    validate_translation_alignment(pl_obj, en_obj, placeholder_allowlist=allowset, strict_allowlist=strict_placeholder_allowlist)
                    sections_en[sec_key] = en_obj
                except ValidationError as ve:
                    failures_en[sec_key] = {"errors": ve.errors, "raw": raw_en[:5000]}
            except JsonParseError as e:
                failures_en[sec_key] = {"errors": [str(e)], "raw": raw_en[:5000]}

        write_json(out / "sections_en.json", sections_en)
        if failures_en:
            write_json(out / "failures_en.json", failures_en)

        # Render English markdown deterministically (headings + deterministic blocks from locale file)
        draft_en = render_markdown(
            doc_type_id=doc.id,
            doc_title=loc_en["title"],
            section_titles=loc_en["sections"],
            sections=sections_en,
            party_lines=loc_en["party_lines"],
            signature_lines=loc_en["signature_lines"],
            safe_sentences=loc_en.get("final_safe_sentences", []),
            append_safe_to_section_7=bool(requirements.get("sections", {}).get("7", {}).get("append_safe_sentences", True)),
        )
        write_text(out / "draft_en.md", draft_en)

    report = {
        "doc_type_id": doc.id,
        "bilingual": bilingual,
        "polish_sections_generated": sorted(list(sections_pl.keys()), key=lambda x: int(x)),
        "polish_failures": failures,
        "english_sections_translated": sorted(list(sections_en.keys()), key=lambda x: int(x)) if bilingual else [],
        "english_failures": failures_en if bilingual else {},
        "rag_enabled": bool(rag_paths),
        "rag_sources_file": "rag_sources.json" if rag_debug else None,
        "legal_warnings_file": "legal_warnings.json",
        "legal_warning_count": legal_warnings.get("warning_count", 0),
        "legal_warning_blockers_file": "legal_warning_blockers.json" if warning_failures else None,
        "legal_warning_blocker_count": len(warning_failures),
    }
    write_json(out / "report.json", report)
    return report
