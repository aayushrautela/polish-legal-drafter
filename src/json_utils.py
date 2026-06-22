from __future__ import annotations

import json
import re
from typing import Any, Optional, Tuple

class JsonParseError(ValueError):
    pass

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", flags=re.IGNORECASE)

def strip_code_fences(text: str) -> str:
    return _CODE_FENCE_RE.sub("", text.strip())

def extract_first_balanced_json_object(text: str) -> Optional[str]:
    """Best-effort extraction of the first balanced JSON object in a string."""
    s = strip_code_fences(text)
    start = s.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(s)):
        ch = s[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return None

def loads_best_effort(text: str) -> Any:
    """Parse a JSON object from text. Raises JsonParseError if parsing fails."""
    s = strip_code_fences(text)

    candidate = extract_first_balanced_json_object(s)
    if candidate is None:
        raise JsonParseError("No JSON object found in model output")

    try:
        return json.loads(candidate)
    except Exception as e:
        raise JsonParseError(f"Failed to parse JSON: {e}\nCandidate: {candidate[:500]}") from e

def dumps_pretty(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)
