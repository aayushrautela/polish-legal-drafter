from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def load_document_spec(path: str | Path) -> Dict[str, Any]:
    spec_path = Path(path)
    if not spec_path.exists():
        raise FileNotFoundError(f"Missing document spec file: {spec_path}")
    return json.loads(spec_path.read_text(encoding="utf-8"))


def load_requirements(requirements_dir: str | Path, doc_type_id: str) -> Dict[str, Any]:
    return load_document_spec(Path(requirements_dir) / f"{doc_type_id}.json")


def requirements_summary(req: Dict[str, Any]) -> str:
    parts = []
    sections = req.get("sections", {})
    for key in sorted(sections.keys(), key=lambda value: int(value)):
        section = sections[key]
        if "min_clauses" in section:
            parts.append(
                f"sekcja {key}: min_clauses={section['min_clauses']}, "
                f"min_words_total={section.get('min_words_total', '-')}"
            )
    return "; ".join(parts)
