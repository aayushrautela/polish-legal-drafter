from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any

@dataclass(frozen=True)
class DocType:
    id: str
    group: str
    title: str
    sections: List[str]

def load_doc_types(path: str | Path) -> Dict[str, DocType]:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    out: Dict[str, DocType] = {}
    for t in cfg.get("types", []):
        out[t["id"]] = DocType(
            id=t["id"],
            group=t.get("group", ""),
            title=t.get("title", t["id"]),
            sections=list(t.get("sections", [])),
        )
    if not out:
        raise ValueError(f"No doc types found in {path}")
    return out
