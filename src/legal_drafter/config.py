"""Configuration for the distillation LLM clients (teacher / checker).

This is the *teacher-side* config: it drives the API models that generate
training data (evidence_reports + draft answers) and the separate *checker*
model that validates it. It is intentionally independent of the student
model (``MODEL_ID`` / ``INFERENCE_ENGINE`` in ``.env.example``), which is
never used to generate its own training data.

Everything is read from the environment (or a ``.env`` file). Two roles are
supported via prefixes:

* ``TEACHER_*``  - generates evidence_reports and draft answers.
* ``CHECKER_*``  - second model that validates outputs (anti-circularity).

OpenAI-compatible: point ``*_BASE_URL`` at any OpenAI-compatible server
(OpenAI, Azure, a local vLLM, etc.). All fields are optional except the
model name.

The module imports only the standard library at top level, so it stays
importable on the public (stdlib-only) path. ``openai`` and ``python-dotenv``
are imported lazily / guarded so their absence is never fatal.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

# Best-effort .env loading; harmless if python-dotenv is not installed or the
# file is absent. Real deployments (secret stores, shell env) work without it.
try:  # pragma: no cover - optional dependency
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass

ROLES = ("teacher", "checker")


def _json_env(name: str) -> Optional[dict]:
    raw = os.environ.get(name)
    if not raw:
        return None
    try:
        val = json.loads(raw)
        return val if isinstance(val, dict) else None
    except (ValueError, TypeError):
        return None


@dataclass
class LLMConfig:
    """Resolved configuration for a single LLM role (teacher or checker)."""

    role: str
    model: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    timeout: float = 120.0
    max_retries: int = 5
    temperature: float = 0.2
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    extra_headers: Optional[dict] = field(default=None)
    extra_body: Optional[dict] = field(default=None)

    @classmethod
    def from_env(cls, role: str = "teacher") -> "LLMConfig":
        if role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}, got {role!r}")
        p = role.upper()

        def get(name: str, default: Optional[str] = None) -> Optional[str]:
            return os.environ.get(f"{p}_{name}", default)

        # The teacher/checker API key falls back to the shared OPENAI_API_KEY
        # so a plain OpenAI setup only needs to set it once.
        api_key = get("API_KEY") or os.environ.get("OPENAI_API_KEY")

        model = get("MODEL")
        if not model:
            raise RuntimeError(
                f"Missing {p}_MODEL in the environment / .env. "
                f"Set {p}_MODEL (e.g. gpt-5) to use the {role} role."
            )

        def opt_int(name: str) -> Optional[int]:
            v = get(name)
            return int(v) if v not in (None, "") else None

        def opt_float(name: str) -> Optional[float]:
            v = get(name)
            return float(v) if v not in (None, "") else None

        return cls(
            role=role,
            model=model,
            base_url=get("BASE_URL"),
            api_key=api_key,
            timeout=opt_float("TIMEOUT") or 120.0,
            max_retries=opt_int("MAX_RETRIES") or 5,
            temperature=opt_float("TEMPERATURE") or 0.2,
            max_tokens=opt_int("MAX_TOKENS"),
            top_p=opt_float("TOP_P"),
            extra_headers=_json_env(f"{p}_EXTRA_HEADERS"),
            extra_body=_json_env(f"{p}_EXTRA_BODY"),
        )

    def masked(self) -> dict:
        """Dict safe to print: the API key is redacted."""
        d = {
            "role": self.role,
            "model": self.model,
            "base_url": self.base_url,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "extra_headers": self.extra_headers,
            "extra_body": self.extra_body,
        }
        d["api_key"] = "***" + (self.api_key[-4:] if self.api_key else "unset")
        return d

    def make_client(self, async_client: bool = False):
        """Build an OpenAI-compatible client from this config.

        ``openai`` is imported lazily so this module stays importable without
        the dependency installed. ``max_retries`` is forwarded to the SDK,
        which already performs exponential backoff on 429/5xx.
        """
        if async_client:
            from openai import AsyncOpenAI

            client_cls = AsyncOpenAI
        else:
            from openai import OpenAI

            client_cls = OpenAI
        return client_cls(
            api_key=self.api_key or "EMPTY",
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )


def load_teacher() -> LLMConfig:
    return LLMConfig.from_env("teacher")


def load_checker() -> LLMConfig:
    return LLMConfig.from_env("checker")


def load_llm_config(role: str) -> LLMConfig:
    return LLMConfig.from_env(role)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Show resolved distillation LLM config.")
    ap.add_argument("--role", choices=ROLES, default="teacher")
    args = ap.parse_args()
    try:
        cfg = LLMConfig.from_env(args.role)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(2)
    print(json.dumps(cfg.masked(), indent=2, ensure_ascii=False))
