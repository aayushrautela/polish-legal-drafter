from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

def load_requirements(requirements_dir: str | Path, doc_type_id: str) -> Dict[str, Any]:
    path = Path(requirements_dir) / f"{doc_type_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing requirements file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))

def requirements_summary(req: Dict[str, Any]) -> str:
    # Keep summary short for prompting
    parts = []
    sections = req.get("sections", {})
    for k in sorted(sections.keys(), key=lambda x: int(x)):
        s = sections[k]
        if "min_clauses" in s:
            parts.append(f"sekcja {k}: min_clauses={s['min_clauses']}, min_words_total={s.get('min_words_total','-')}")
    return "; ".join(parts)
