from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path


def run_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    value = re.sub(r"-+", "-", value).strip("-._")
    return value or "run"


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(2, 1000):
        candidate = path.with_name(f"{path.name}-{index}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not create unique path for {path}")


def timestamped_child(parent: Path, label: str, timestamp: str | None = None) -> Path:
    return unique_path(parent / f"{timestamp or run_timestamp()}_{slugify(label)}")


def resolve_output_path(project_root: Path, requested: str, default_parent: str, label: str) -> Path:
    if requested:
        path = Path(requested)
        return path if path.is_absolute() else project_root / path
    return timestamped_child(project_root / default_parent, label)


def relative_to_root(project_root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)
