from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

def load_locale(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))

def render_markdown(
    doc_type_id: str,
    doc_title: str,
    section_titles: List[str],
    sections: Dict[str, Dict[str, Any]],
    party_lines: List[str],
    signature_lines: List[str],
    safe_sentences: List[str],
    append_safe_to_section_7: bool = True,
) -> str:
    lines: List[str] = []
    lines.append(f"# {doc_title}")
    lines.append("")

    for idx, sec_title in enumerate(section_titles, start=1):
        lines.append(f"## {idx}. {sec_title}")
        if idx == 1:
            # deterministic party block
            lines.extend(party_lines)
            lines.append("")
            continue

        if sec_title.lower() in ("podpisy", "podpis"):
            # deterministic signatures
            lines.extend(signature_lines)
            lines.append("")
            continue

        sec_obj = sections.get(str(idx))
        if not sec_obj:
            lines.append("[[ERROR: missing section content]]")
            lines.append("")
            continue

        clauses = sec_obj.get("clauses", [])
        for c in clauses:
            cid = c.get("id")
            txt = c.get("text", "").strip()
            if cid and txt:
                lines.append(f"{cid}. {txt}")
                lines.append("")  # blank line between clauses

        if append_safe_to_section_7 and idx == 7 and safe_sentences:
            for s in safe_sentences:
                lines.append(s)
            lines.append("")

    return "\n".join(lines).strip() + "\n"
