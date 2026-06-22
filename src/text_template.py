from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

@dataclass(frozen=True)
class TemplateVars:
    values: Dict[str, str]

def render_template(template: str, vars: Dict[str, str]) -> str:
    """Replace {{VARNAME}} occurrences with provided string values.

    This avoids Python's str.format() curly-brace conflicts (useful because prompts often contain JSON).
    """
    out = template
    for k, v in vars.items():
        out = out.replace("{{" + k + "}}", v)
    return out
