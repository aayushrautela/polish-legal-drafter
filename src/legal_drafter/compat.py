"""Single OpenAI-shape <-> Anthropic native API compatibility layer.

The project talks to every teacher/checker/judge through the OpenAI
``chat.completions`` shape. Anthropic's *OpenAI-compatible* surface is a thin
proxy that silently drops two things our distillation pipeline depends on:

* ``reasoning_content`` -- the OpenAI-compat endpoint never returns Claude's
  thinking trace (Anthropic documents this explicitly).
* ``response_format`` / ``strict`` on tools -- both are **ignored**, so the
  strict ``{"summary", "contract"}`` envelope is never enforced.

This module is the *one* translation layer that fixes both by calling
Anthropic's native ``/v1/messages`` API instead of the compat proxy, while
still presenting an OpenAI-shaped client so the runners don't change shape.

Design decisions (kept deliberately small -- see issue analysis):
* **Non-streaming upstream.** Our ``_stream_turn`` consumes any iterable of
  chunks; the adapter calls the native API without ``stream`` and yields a
  single synthetic chunk. This removes the entire SSE-translation problem
  (~400 lines avoided vs LiteLLM).
* **Thinking round-trip.** ``thinking.type`` is ``adaptive`` with
  ``display: "summarized"`` so a real trace comes back as ``reasoning_content``.
  Signed thinking blocks received in a prior assistant turn are passed straight
  back (Anthropic verifies signatures); unsigned ones (which we only ever have
  as concatenated ``reasoning_content`` text) are dropped, and adaptive mode
  degrades gracefully without a 400.
* **Envelope fix.** ``response_format`` (json_schema / json_object) is mapped to
  native ``output_config.format`` -- constrained decoding, so the contract
  envelope is guaranteed at the token level. This is the proper fix the
  OpenAI-compat layer cannot do.
* **Sampling policy lives in ``llm_params``.** This layer only applies
  Anthropic-native constraints (temperature is locked to 1 while thinking is
  on). What to send is decided upstream by ``llm_params.chat_create``.

Request hygiene (empty blocks, message alternation, trailing whitespace) is
handled here because the native API is strict about it.
"""

from __future__ import annotations

import json
import os
import types
import warnings
from typing import Any, Iterator, Mapping, Optional

__all__ = [
    "AnthropicCompatClient",
    "CompatAPIError",
    "make_chat_client",
    "compat_mode",
    "build_anthropic_request",
    "translate_response",
]

ANTHROPIC_VERSION = "2023-06-01"
_DEFAULT_BASE_URL = "https://api.anthropic.com"
_DEFAULT_MAX_TOKENS = 8192

# Native stop_reason -> OpenAI finish_reason
_STOP_REASON_MAP = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "pause_turn": "stop",
    "max_tokens": "length",
    "model_context_window_exceeded": "length",
    "tool_use": "tool_calls",
    "refusal": "content_filter",
}


def _ns(**kw) -> types.SimpleNamespace:
    return types.SimpleNamespace(**kw)


def _text_of(content: Any) -> str:
    """Flatten OpenAI-shaped message content to a string for Anthropic."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for p in content:
            if isinstance(p, dict):
                if p.get("type") == "text":
                    parts.append(str(p.get("text") or ""))
                elif p.get("type") in ("image_url", "image"):
                    raise ValueError(
                        "image content is not supported by the Anthropic compat layer")
            elif isinstance(p, str):
                parts.append(p)
        return "".join(parts)
    return str(content)


def _tool_use_input(arguments: Any) -> dict:
    if not arguments:
        return {}
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            obj = json.loads(arguments)
            return obj if isinstance(obj, dict) else {"_raw": obj}
        except Exception:
            return {"_raw": arguments}
    return {"_raw": arguments}


def _merge(out: list, role: str, blocks: list) -> None:
    if out and out[-1]["role"] == role:
        out[-1]["content"].extend(blocks)
    else:
        out.append({"role": role, "content": blocks})


def translate_messages(messages: list) -> tuple[list, list[str]]:
    """OpenAI messages -> (anthropic_messages, system_text_blocks)."""
    system: list[str] = []
    out: list[dict] = []
    for m in messages or []:
        role = m.get("role")
        if role == "system":
            t = _text_of(m.get("content"))
            if t.strip():
                system.append(t)
        elif role == "user":
            t = _text_of(m.get("content"))
            if t.strip():
                _merge(out, "user", [{"type": "text", "text": t}])
        elif role == "assistant":
            blocks: list[dict] = []
            for tb in m.get("thinking_blocks") or []:
                if isinstance(tb, dict) and str(tb.get("signature") or "").strip():
                    blocks.append(dict(tb))
            t = _text_of(m.get("content")).rstrip()
            if t:
                blocks.append({"type": "text", "text": t})
            for i, tc in enumerate(m.get("tool_calls") or []):
                fn = tc.get("function") or {}
                blocks.append({
                    "type": "tool_use",
                    "id": tc.get("id") or f"toolu_{i}",
                    "name": fn.get("name"),
                    "input": _tool_use_input(fn.get("arguments")),
                })
            if not blocks:
                blocks = [{"type": "text", "text": "."}]
            _merge(out, "assistant", blocks)
        elif role in ("tool", "function"):
            content = m.get("content")
            text = content if isinstance(content, str) else \
                json.dumps(content, ensure_ascii=False)
            _merge(out, "user", [{
                "type": "tool_result",
                "tool_use_id": str(m.get("tool_call_id") or ""),
                "content": text,
            }])
        else:
            raise ValueError(f"unsupported message role: {role!r}")
    # Anthropic rejects empty text content blocks ("text content blocks must
    # be non-empty") and requires at least one message. A turn whose content
    # vanished (e.g. an empty user message that contributed nothing to a
    # merge) gets a placeholder; a fully empty conversation gets one turn.
    for msg in out:
        if not msg["content"]:
            msg["content"] = [{"type": "text", "text": "."}]
    if not out:
        out.append({"role": "user",
                    "content": [{"type": "text", "text": "."}]})
    return out, system


def translate_tools(tools: Optional[list]) -> list:
    out: list[dict] = []
    for t in tools or []:
        fn = (t or {}).get("function") or {}
        name = fn.get("name")
        if not name:
            continue
        tool: dict = {"name": name,
                      "input_schema": fn.get("parameters") or {"type": "object"}}
        if fn.get("description"):
            tool["description"] = fn["description"]
        out.append(tool)
    return out


def translate_tool_choice(tc: Any) -> Optional[dict]:
    if tc is None:
        return None
    if tc == "auto":
        return {"type": "auto"}
    if tc == "none":
        return {"type": "none"}
    if tc == "required":
        return {"type": "any"}
    if isinstance(tc, dict):
        if tc.get("type") == "function":
            name = (tc.get("function") or {}).get("name")
            return {"type": "tool", "name": name} if name else {"type": "auto"}
    return None


def translate_response_format(rf: Any) -> Optional[dict]:
    if not isinstance(rf, dict):
        return None
    t = rf.get("type")
    if t == "json_schema":
        js = rf.get("json_schema") or {}
        schema = js.get("schema") or rf.get("schema")
        if isinstance(schema, dict):
            return {"type": "json_schema", "schema": schema}
    if t == "json_object":
        return {"type": "json_schema", "schema": {"type": "object"}}
    return None


def build_anthropic_request(
    kwargs: Mapping[str, Any],
    *,
    thinking_mode: str,
    default_max_tokens: int,
) -> tuple[dict, list[str]]:
    """Build the native ``/v1/messages`` body. Returns (body, notes[]).

    ``notes`` records any Anthropic-native adjustments (e.g. temperature
    dropped because thinking locks it to 1) for provenance/manifests.
    """
    notes: list[str] = []
    messages = kwargs.get("messages") or []
    anthropic_messages, system = translate_messages(messages)
    max_tokens = kwargs.get("max_tokens") or kwargs.get("max_completion_tokens") \
        or default_max_tokens
    body: dict = {
        "model": kwargs.get("model"),
        "max_tokens": int(max_tokens),
        "messages": anthropic_messages,
    }
    if system:
        body["system"] = [{"type": "text", "text": s} for s in system]

    # Sampling passthrough. llm_params.py owns the send/drop policy; this layer
    # only applies Anthropic-native constraints.
    for key in ("temperature", "top_p"):
        val = kwargs.get(key)
        if val is None:
            continue
        if (thinking_mode != "off" and key == "temperature"
                and float(val) != 1.0):
            notes.append(
                f"dropped {key}={val} (Anthropic locks temperature to 1 "
                f"while thinking is enabled)")
            continue
        body[key] = val

    stop = kwargs.get("stop")
    if stop:
        body["stop_sequences"] = [stop] if isinstance(stop, str) else list(stop)

    tools = translate_tools(kwargs.get("tools"))
    if tools:
        body["tools"] = tools
    tch = translate_tool_choice(kwargs.get("tool_choice"))
    if tch:
        body["tool_choice"] = tch

    oc = translate_response_format(kwargs.get("response_format"))
    if oc:
        body["output_config"] = {"format": oc}

    if thinking_mode != "off":
        body["thinking"] = {"type": thinking_mode, "display": "summarized"}

    extra = kwargs.get("extra_body")
    if isinstance(extra, dict):
        body.update(extra)
    return body, notes


def translate_response(data: dict) -> dict:
    """Native ``/v1/messages`` JSON -> OpenAI chat.completion-shaped pieces."""
    text, reasoning, thinking_blocks, tool_calls = "", "", [], []
    for i, block in enumerate(data.get("content") or []):
        btype = block.get("type")
        if btype == "text":
            text += block.get("text") or ""
        elif btype == "thinking":
            reasoning += block.get("thinking") or ""
            thinking_blocks.append(block)
        elif btype == "redacted_thinking":
            thinking_blocks.append(block)
        elif btype == "tool_use":
            tool_calls.append({
                "index": i,
                "id": block.get("id") or f"call_{i}",
                "type": "function",
                "function": {
                    "name": block.get("name"),
                    "arguments": json.dumps(block.get("input") or {},
                                             ensure_ascii=False),
                },
            })

    usage = data.get("usage") or {}
    cached = int(usage.get("cache_read_input_tokens") or 0)
    cache_create = int(usage.get("cache_creation_input_tokens") or 0)
    prompt = int(usage.get("input_tokens") or 0) + cached + cache_create
    completion = int(usage.get("output_tokens") or 0)

    return {
        "model": data.get("model"),
        "id": f"chatcmpl-{str(data.get('id') or '')}",
        "text": text,
        "reasoning": reasoning,
        "thinking_blocks": thinking_blocks,
        "tool_calls": tool_calls,
        "finish_reason": _STOP_REASON_MAP.get(data.get("stop_reason"), "stop"),
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
            "cached_tokens": cached,
        },
    }


class CompatAPIError(Exception):
    """Raised on non-200 from the Anthropic API.

    Carries ``status_code`` so ``llm_params._is_sampling_rejection`` can detect
    sampling-parameter 400s and retry without them.
    """

    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.message = message
        self.status_code = int(status_code)


class AnthropicCompatClient:
    """OpenAI-shaped client that calls Anthropic's native ``/v1/messages``."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        *,
        thinking_mode: Optional[str] = None,
        default_max_tokens: Optional[int] = None,
        timeout: float = 600.0,
        _transport: Any = None,
    ):
        self.base_url = (base_url or os.environ.get("ANTHROPIC_BASE_URL")
                         or _DEFAULT_BASE_URL).rstrip("/")
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY") or ""
        env_mode = (os.environ.get("ANTHROPIC_COMPAT_THINKING") or "").strip().lower()
        self.thinking_mode = thinking_mode or env_mode or "adaptive"
        self.default_max_tokens = int(
            default_max_tokens
            or os.environ.get("ANTHROPIC_COMPAT_MAX_TOKENS")
            or _DEFAULT_MAX_TOKENS)
        import httpx
        self._http = httpx.Client(transport=_transport, timeout=timeout) \
            if _transport else httpx.Client(timeout=timeout)
        self.chat = _ns(completions=_ns(create=self.create))
        self.last_request: Optional[dict] = None
        self.last_notes: list[str] = []

    def _url(self) -> str:
        return self.base_url + ("/messages"
                                if self.base_url.endswith("/v1")
                                else "/v1/messages")

    def _headers(self) -> dict:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }

    def _post(self, body: dict) -> dict:
        resp = self._http.post(self._url(), json=body, headers=self._headers())
        if resp.status_code != 200:
            try:
                err = resp.json()
                msg = (err.get("error") or {}).get("message") or resp.text[:500]
            except Exception:
                msg = resp.text[:500]
            raise CompatAPIError(str(msg), resp.status_code)
        return resp.json()

    def create(self, *, stream: bool = False, **kwargs: Any) -> Any:
        body, notes = build_anthropic_request(
            kwargs, thinking_mode=self.thinking_mode,
            default_max_tokens=self.default_max_tokens)
        self.last_request = body
        self.last_notes = notes
        if notes:
            warnings.warn(
                "[anthropic-compat] " + "; ".join(notes), stacklevel=3)
        data = self._post(body)
        pieces = translate_response(data)
        if stream:
            return iter([self._chunk(pieces)])
        return self._response(pieces)

    def _usage_ns(self, usage: dict) -> types.SimpleNamespace:
        return _ns(
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            total_tokens=usage["total_tokens"],
            cached_tokens=usage.get("cached_tokens", 0),
        )

    def _chunk(self, p: dict) -> types.SimpleNamespace:
        delta = _ns(
            content=p["text"] or None,
            reasoning=p["reasoning"] or None,
            reasoning_content=p["reasoning"] or None,
            tool_calls=[
                _ns(index=tc["index"], id=tc["id"], type="function",
                    function=_ns(name=tc["function"]["name"],
                                 arguments=tc["function"]["arguments"]))
                for tc in p["tool_calls"]
            ] or None,
        )
        return _ns(
            id=p["id"], object="chat.completion.chunk",
            choices=[_ns(index=0, delta=delta, finish_reason=p["finish_reason"])],
            usage=self._usage_ns(p["usage"]),
        )

    def _response(self, p: dict) -> types.SimpleNamespace:
        message = _ns(
            role="assistant",
            content=p["text"] or None,
            reasoning_content=p["reasoning"] or None,
            reasoning=p["reasoning"] or None,
            tool_calls=[
                _ns(id=tc["id"], type="function",
                    function=_ns(name=tc["function"]["name"],
                                 arguments=tc["function"]["arguments"]))
                for tc in p["tool_calls"]
            ] or None,
            thinking_blocks=p["thinking_blocks"],
        )
        return _ns(
            id=p["id"], model=p["model"], object="chat.completion",
            choices=[_ns(index=0, message=message,
                        finish_reason=p["finish_reason"])],
            usage=self._usage_ns(p["usage"]),
        )


# --------------------------------------------------------------------------
# Construction / factory
# --------------------------------------------------------------------------

def _lookup(key: str, env: Optional[Mapping[str, str]]) -> str:
    if env:
        v = env.get(key)
        if v:
            return str(v)
    v = os.environ.get(key)
    return v or ""


def _resolve_compat(role: str, env: Optional[Mapping[str, str]]) -> str:
    for key in (f"{role.upper()}_COMPAT", "LLM_COMPAT"):
        v = _lookup(key, env).strip().lower()
        if v:
            return v
    return ""


def compat_mode(role: str, env: Optional[Mapping[str, str]] = None) -> str:
    """Return ``"anthropic"`` or ``"openai"`` for a role."""
    return "anthropic" if _resolve_compat(role, env) in ("anthropic", "claude") \
        else "openai"


def make_chat_client(
    role: str,
    base_url: str,
    api_key: str,
    *,
    env: Optional[Mapping[str, str]] = None,
    timeout: Optional[float] = None,
    max_retries: Optional[int] = None,
) -> Any:
    """Sole client-construction entry point.

    Reads ``<ROLE>_COMPAT`` / ``LLM_COMPAT`` (env dict first, then
    os.environ). When set to ``anthropic``/``claude`` returns an
    :class:`AnthropicCompatClient`; otherwise the normal OpenAI SDK client.
    """
    compat = _resolve_compat(role, env)
    if compat in ("anthropic", "claude"):
        return AnthropicCompatClient(
            base_url=_lookup("ANTHROPIC_BASE_URL", env) or base_url,
            api_key=_lookup("ANTHROPIC_API_KEY", env) or api_key,
        )
    from openai import OpenAI
    kw: dict = {}
    if timeout is not None:
        kw["timeout"] = timeout
    if max_retries is not None:
        kw["max_retries"] = max_retries
    return OpenAI(base_url=base_url, api_key=api_key, **kw)
