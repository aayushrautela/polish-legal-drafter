from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .doc_types import DocType, load_doc_types
from .document_specs import load_document_spec
from .io_utils import read_json, read_text, write_json, write_text
from .pipeline import load_prompts
from .renderer import load_locale, render_markdown

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOC_TYPES = PROJECT_ROOT / "config" / "doc_types" / "public_doc_types.json"
DEFAULT_DOCUMENT_SPECS = PROJECT_ROOT / "config" / "document_specs"
DEFAULT_PROMPTS = PROJECT_ROOT / "config" / "prompts"
DEFAULT_LOCALES = PROJECT_ROOT / "config" / "locales"
DEFAULT_SCHEMAS = PROJECT_ROOT / "config" / "schemas"


def _load_supported_doc(doc_type_id: str, doc_types_path: Path) -> DocType:
    doc_types = load_doc_types(doc_types_path)
    if doc_type_id not in doc_types:
        supported = ", ".join(sorted(doc_types))
        raise ValueError(f"Unsupported doc_type_id={doc_type_id}. Supported: {supported}")
    return doc_types[doc_type_id]


def _sample_clause(section_number: int, section_title: str, topics: list[str], min_clauses: int) -> dict[str, Any]:
    clauses = []
    topic_text = ", ".join(topics[:3]) if topics else section_title.lower()
    for idx in range(1, min(max(min_clauses, 1), 3) + 1):
        clauses.append({
            "id": f"{section_number}.{idx}",
            "text": (
                f"Strony ustalają zasady dotyczące: {topic_text}. "
                "Szczegółowe dane pozostają oznaczone właściwymi placeholderami i wymagają uzupełnienia przed użyciem dokumentu."
            ),
            "source_ids": [],
        })
    return {
        "section_number": section_number,
        "section_title": section_title,
        "clauses": clauses,
    }


def run_smoke_draft(
    *,
    facts_file: Path,
    out_dir: Path,
    doc_type_id: str = "UMOWA_ZLECENIA",
    doc_types_path: Path = DEFAULT_DOC_TYPES,
    document_specs_dir: Path = DEFAULT_DOCUMENT_SPECS,
    prompts_dir: Path = DEFAULT_PROMPTS,
    locales_dir: Path = DEFAULT_LOCALES,
    schemas_dir: Path = DEFAULT_SCHEMAS,
) -> dict[str, Any]:
    doc = _load_supported_doc(doc_type_id, doc_types_path)
    spec = load_document_spec(document_specs_dir / "umowa_zlecenia.v1.json")
    placeholders = read_json(schemas_dir / "placeholders.json")
    facts = read_text(facts_file)
    prompts = load_prompts(prompts_dir)
    locale_pl = load_locale(locales_dir / "pl.json")[doc.id]

    sections: dict[str, dict[str, Any]] = {}
    for idx, section_title in enumerate(doc.sections, start=1):
        section_spec = spec.get("sections", {}).get(str(idx), {})
        if section_spec.get("mode") in {"deterministic_party_block", "deterministic_signature_block"}:
            continue
        sections[str(idx)] = _sample_clause(
            idx,
            section_title,
            list(section_spec.get("topics", [])),
            int(section_spec.get("min_clauses", 1)),
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    draft = render_markdown(
        doc_type_id=doc.id,
        doc_title=locale_pl["title"],
        section_titles=locale_pl["sections"],
        sections=sections,
        party_lines=locale_pl["party_lines"],
        signature_lines=locale_pl["signature_lines"],
        safe_sentences=locale_pl.get("final_safe_sentences", []),
    )
    report = {
        "mode": "smoke",
        "doc_type_id": doc.id,
        "facts_file": str(facts_file),
        "facts_characters": len(facts),
        "prompt_templates_loaded": sorted(prompts.__dict__.keys()),
        "placeholder_count": len(placeholders.get("allowlist", [])),
        "draft_file": "draft_pl.md",
    }
    write_text(out_dir / "draft_pl.md", draft)
    write_json(out_dir / "report.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="legal-drafter")
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("smoke", help="Run a deterministic local smoke draft")
    smoke.add_argument("--facts-file", type=Path, default=PROJECT_ROOT / "examples" / "facts_umowa_zlecenia.txt")
    smoke.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "outputs" / "smoke")
    smoke.add_argument("--doc-type", default="UMOWA_ZLECENIA")

    validate = subparsers.add_parser("validate-config", help="Validate public configuration files can be loaded")
    validate.add_argument("--doc-type", default="UMOWA_ZLECENIA")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "smoke":
        report = run_smoke_draft(facts_file=args.facts_file, out_dir=args.out_dir, doc_type_id=args.doc_type)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    if args.command == "validate-config":
        doc = _load_supported_doc(args.doc_type, DEFAULT_DOC_TYPES)
        load_document_spec(DEFAULT_DOCUMENT_SPECS / "umowa_zlecenia.v1.json")
        load_prompts(DEFAULT_PROMPTS)
        load_locale(DEFAULT_LOCALES / "pl.json")
        read_json(DEFAULT_SCHEMAS / "placeholders.json")
        print(json.dumps({"status": "ok", "doc_type_id": doc.id}, ensure_ascii=False))
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
