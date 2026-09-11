"""OpenAI-compatible chat-completion request hygiene.

Newer Claude models (Sonnet 5, Opus 4.7/4.8/5, Fable 5, Mythos 5) reject
non-default sampling parameters (``temperature``/``top_p``/``top_k``) with
HTTP 400 ("temperature is deprecated for this model") — including over the
OpenAI-compatible endpoint. Anthropic's guidance is to omit the parameters
entirely rather than retry on 400.

Because models are usually reached through router aliases (the real model ID
is invisible to the client), this module does NOT sniff model names. Instead:

1. ``*_SAMPLING`` / ``LLM_SAMPLING`` env knobs let the operator declare the
   mode per role: ``on`` (default, parameters sent as requested) or ``off``
   (sampling parameters never sent).
2. Values equal to the provider default (1.0) are never sent — identical
   semantics on every OpenAI-compatible backend.
3. As a safety net, a 400 whose message blames a sampling parameter triggers
   exactly one retry without it, and the model alias is cached for the rest
   of the process (no perpetual retry path).

Every request's requested-vs-sent decision is recorded in ``DECISIONS`` so
run manifests can show what actually reached the API.
"""

from __future__ import annotations

import os
import warnings
from typing import Any

__all__ = [
    "sampling_mode",
    "resolve_sampling_params",
    "chat_create",
    "sampling_report",
    "DECISIONS",
]

_MODES = ("on", "off")

# (role, model) -> {"requested": {...}, "sent": {...}, "mode": str}
DECISIONS: dict[tuple[str, str], dict[str, Any]] = {}

# Model aliases proven to reject sampling params (filled by 400 detection).
_STRIP_CACHE: set[str] = set()

# Warn at most once per (role, model, kind) to keep logs readable under
# high-concurrency runs.
_WARNED: set[tuple[str, str, str]] = set()

_SAMPLING_KEYS = ("temperature", "top_p", "top_k")


def _env_mode(name: str) -> str | None:
    raw = (os.environ.get(name) or "").strip().lower()
    return raw if raw in _MODES else None


def sampling_mode(role: str) -> str:
    """Resolve the sampling mode for a role.

    Precedence: ``<ROLE>_SAMPLING`` > global ``LLM_SAMPLING`` > ``"on"``.
    Unknown values fall through (never crash a run over a typo'd env var).
    """
    return _env_mode(f"{role.upper()}_SAMPLING") \
        or _env_mode("LLM_SAMPLING") \
        or "on"


def _warn_once(role: str, model: str, kind: str, message: str) -> None:
    key = (role, model, kind)
    if key in _WARNED:
        return
    _WARNED.add(key)
    warnings.warn(f"[llm_params role={role} model={model}] {message}",
                  stacklevel=3)


def resolve_sampling_params(role: str, model: str, *,
                            temperature: float | None = None,
                            top_p: float | None = None,
                            top_k: float | None = None) -> dict[str, float]:
    """Decide which sampling parameters may be sent for this request.

    Returns only the keys that should appear in the request body. Never
    raises; unknown env values degrade to mode ``"on"``.
    """
    requested = {k: v for k, v in
                 (("temperature", temperature), ("top_p", top_p),
                  ("top_k", top_k))
                 if v is not None}
    mode = sampling_mode(role)
    sent: dict[str, float] = {}

    if mode == "off":
        if requested:
            _warn_once(
                role, model, "off",
                f"{role.upper()}_SAMPLING=off: dropping requested sampling "
                f"params {requested}; diversity intent will NOT reach the API.")
    elif model in _STRIP_CACHE:
        if requested:
            _warn_once(
                role, model, "cached",
                f"model previously rejected sampling params (HTTP 400); "
                f"dropping {requested}. Set {role.upper()}_SAMPLING=off to "
                f"skip the failed attempt on future runs.")
    else:
        # Values equal to the provider default are semantically identical
        # to omission on every OpenAI-compatible backend -> never send them.
        sent = {k: v for k, v in requested.items() if float(v) != 1.0}

    DECISIONS[(role, model)] = {
        "requested": dict(requested),
        "sent": dict(sent),
        "mode": mode,
        "stripped_cached": model in _STRIP_CACHE,
    }
    return sent


def _is_sampling_rejection(exc: Exception) -> bool:
    """True for HTTP 400s whose message blames a sampling parameter."""
    if getattr(exc, "status_code", None) != 400:
        return False
    msg = str(exc).lower()
    names = ("temperature", "top_p", "top_k")
    reasons = ("deprecated", "not supported", "unsupported")
    return any(n in msg for n in names) and any(r in msg for r in reasons)


def chat_create(client: Any, *, role: str, **kwargs: Any) -> Any:
    """``client.chat.completions.create`` with sampling-param compatibility.

    Extracts ``temperature``/``top_p``/``top_k`` from ``kwargs``, resolves
    what may actually be sent via :func:`resolve_sampling_params`, and
    delegates. On a sampling-parameter HTTP 400, retries exactly once with
    the parameters removed and caches the model alias (subsequent calls in
    this process skip straight to the clean payload). All other errors —
    including unrelated 400s — propagate unchanged.
    """
    model = str(kwargs.get("model", ""))
    requested = {k: kwargs.pop(k) for k in _SAMPLING_KEYS if k in kwargs}
    sent = resolve_sampling_params(role, model, **requested)
    kwargs.update(sent)
    try:
        return client.chat.completions.create(**kwargs)
    except Exception as exc:
        if not _is_sampling_rejection(exc) or not sent:
            raise
        for k in _SAMPLING_KEYS:
            kwargs.pop(k, None)
        _STRIP_CACHE.add(model)
        _warn_once(
            role, model, "detected",
            f"HTTP 400 rejected sampling params {sent} "
            f"({type(exc).__name__}: {str(exc)[:200]}); retrying once without "
            f"them and caching the alias. Set {role.upper()}_SAMPLING=off to "
            f"avoid the failed attempt on future runs.")
        return client.chat.completions.create(**kwargs)


def sampling_report(role: str, model: str) -> dict[str, Any] | None:
    """Last requested-vs-sent decision for a (role, model) pair, or None."""
    return DECISIONS.get((role, model))
