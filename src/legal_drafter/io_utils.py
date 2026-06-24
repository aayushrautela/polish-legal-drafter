from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

def read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")

def write_text(path: str | Path, text: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")

def read_json(path: str | Path) -> Any:
    return json.loads(read_text(path))

def write_json(path: str | Path, obj: Any) -> None:
    write_text(path, json.dumps(obj, ensure_ascii=False, indent=2) + "\n")

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
