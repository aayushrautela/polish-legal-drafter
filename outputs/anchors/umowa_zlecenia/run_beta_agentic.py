"""PHASE BETA v2 - TRUE agentic template editing (user's original vision).

Each independent run: AI is told to FIRST read the template via get_template
tool, THEN pull whatever grounding material IT decides via semantic_search /
keyword_search / chunk_read, and only then emit structured add/remove/modify
edits. Full tool-call transcript recorded per run so we can SEE what it chose
to retrieve and what it added/removed.

Retrieval stays on Modal (set_remote -> GPU endpoint). Streaming per agent-1's
make_chat pattern. Three runs in parallel, independent sessions.
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / ".." / ".." / ".." / "src"))

from openai import OpenAI                                    # noqa: E402
from legal_drafter.retrieval.agentic_tools import (          # noqa: E402
    TOOL_SCHEMAS, execute_tool, set_remote,
)
from legal_drafter.retrieval.remote import RemoteRetrieval   # noqa: E402

ENDPOINT = "https://nikkybrekas--legal-drafter-retrieval-retrievalservicegpu-app.modal.run"
MAX_ITER = 6

SYSTEM = (
    "Jesteś doświadczonym polskim prawnikiem-redaktorem. Twoje zadanie: "
    "rozszerzyć wzór UMOWY ZLECENIA o dodatkowe postanowienia dla konkretnych "
    "sytuacji klientów. MASZ NARZĘDZIA RETRIEVAL: najpierw przeczytaj wzór "
    "(get_template), potem sam zdecyduj, jakie materiały prawne i klauzule "
    "pobrać z korpusu (semantic_search / keyword_search / chunk_read), żeby "
    "edycje były poprawne prawnie i osadzone w prawdziwych źródłach. Pracuj "
    "iteracyjnie: szukaj -> czytaj -> decyduj. "
    "WAŻNE: W JEDNEJ TURZE WYWOŁAJ DOKŁADNIE JEDNO NARZĘDZIE."
)

FORCED_LAST = (
    "TO JEST OSTATECZNY KROK. Koniec wyszukiwania. Zwróć WYŁĄCZNIE strict JSON "
    "(bez markdown): {\"edits\":[{\"op\":\"add|remove|modify\","
    "\"target\":\"<sekcja lub fraza z wzoru>\",\"new_text\":\"<pełna polska "
    "klauzula do wklejenia>\",\"old_text\":\"<fragment do usunięcia/zmiany, "
    "puste dla add>\",\"cite_ref\":\"<chunk_id/source_ref użytego materiału>\","
    "\"reason\":\"<jedno zdanie>\"}]} - 3 do 6 edycji."
)


def load_env(path=".env"):
    env = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env


def extract_json(text):
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```", 2)[1]
        if t.startswith("json"):
            t = t[4:]
    m = re.search(r"\{.*\}", t, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _parse_text_tag_args(raw: str) -> str:
    """Parse Qwen/Hermes-style text-tag tool args that some providers
    (KiloCode/tencent-hy3) pass through unconverted in the arguments field.
    Extracts <arg_key>X</arg_key><arg_value>Y</arg_value> pairs into proper
    JSON. Falls back to raw_decode for standard JSON args."""
    import re
    raw_stripped = (raw or "").strip()
    # try standard JSON first
    try:
        obj = json.loads(raw_stripped)
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        pass
    # text-tag format: extract <arg_key>name</arg_key>...<arg_value>val</arg_value>
    keys = re.findall(r"<arg_key[^>]*>(.*?)</arg_key", raw_stripped)
    vals = re.findall(r"<arg_value[^>]*>(.*?)</arg_value", raw_stripped)
    if keys and vals and len(keys) == len(vals):
        obj = {}
        for key, val in zip(keys, vals):
            key = key.strip()
            val = val.strip()
            try:
                val = json.loads(val)
            except Exception:
                pass
            obj[key] = val
        return json.dumps(obj, ensure_ascii=False)
    # raw_decode fallback (litellm #20480 doubled JSON)
    try:
        dec = json.JSONDecoder()
        obj, _ = dec.raw_decode(raw_stripped)
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        pass
    # last resort: if there's a {..} somewhere, extract it
    start = raw_stripped.find("{")
    if start >= 0:
        try:
            obj, _ = dec.raw_decode(raw_stripped[start:])
            return json.dumps(obj, ensure_ascii=False)
        except Exception:
            pass
    return "{}"


def _sanitize_tool_calls(calls, session_key: str):
    """Fix corrupted tool calls from providers with broken FC bridges:
    - text-tag args (Qwen/Hermes format passed through unconverted)
    - litellm #20480 doubled-summary-args
    - empty/duplicate ids
    Also strips empty content from assistant messages (opencode pattern)."""
    out, seen = [], set()
    for k, tc in enumerate(calls):
        name = tc.function.name or f"tool_{k}"
        raw = (tc.function.arguments or "")
        args = _parse_text_tag_args(raw)
        cid = tc.id or ""
        if not cid or cid in seen:
            cid = f"call_{session_key}_{k}_{abs(hash(args)) % 99999}"
        seen.add(cid)
        out.append(types.SimpleNamespace(
            id=cid, type="function",
            function=types.SimpleNamespace(name=name, arguments=args)))
    return out


def make_stream_chat(client, model, session_id, temperature=0.2):
    """Agent-1 style streaming turn WITH tools declared on EVERY request.
    Fixes vs first pilot: (a) tools=TOOL_SCHEMAS actually sent - without it
    the omnirouter injected its own Exa web-search tools; (b) litellm
    session id groups the conversation; (c) tool-call deltas accumulated by
    tc.id, NOT index (litellm #21331 collapses parallel indices to 0)."""

    _turn = {"n": 0}
    def chat(messages):
        # STREAMING + tools; payload dumped per-turn so a 400 can be bisected
        # against providers that work (opencode/codex drive this same route).
        _turn["n"] += 1
        tweak = os.environ.get("BETA_TWEAK", "")
        kwargs = dict(model=model, messages=messages, temperature=temperature,
                      max_tokens=int(os.environ.get("BETA_MAX_TOKENS", "16000")),
                      timeout=600, stream=True,
                      tools=TOOL_SCHEMAS, tool_choice="auto")
        if tweak == "nochoice":
            kwargs.pop("tool_choice")
        if tweak == "notemp":
            kwargs.pop("temperature")
        if tweak == "nomax":
            kwargs.pop("max_tokens")
        if tweak == "contentnull":
            cleaned = []
            for m in messages:
                m = dict(m)
                if m.get("role") == "assistant" and m.get("tool_calls") and not (m.get("content") or "").strip():
                    m.pop("content", None)
                cleaned.append(m)
            kwargs["messages"] = cleaned
        Path(f"{HERE}/beta_experiment/last_request_t{_turn['n']}.json").write_text(
            json.dumps(kwargs, ensure_ascii=False, default=str), encoding="utf-8")
        parts, tc_map, vidx = [], {}, {}
        next_vidx = 0
        for chunk in client.chat.completions.create(**kwargs):
            if not chunk.choices:
                continue
            d = chunk.choices[0].delta
            if d is None:
                continue
            if getattr(d, "content", None):
                parts.append(d.content)
            for tc in getattr(d, "tool_calls", None) or []:
                if tc.id:
                    v = vidx.setdefault(tc.id, len(vidx))
                else:
                    v = vidx.get(tc.index, next_vidx)
                slot = tc_map.setdefault(v, {"id": "", "name": "", "args": ""})
                if getattr(tc, "index", 0) not in (None,) and tc.id is None:
                    next_vidx = max(next_vidx, v + 1)
                if tc.id:
                    slot["id"] += tc.id
                if getattr(tc.function, "name", None):
                    slot["name"] += tc.function.name
                if getattr(tc.function, "arguments", None):
                    slot["args"] += tc.function.arguments
        clean = [types.SimpleNamespace(
                    id=s["id"], type="function",
                    function=types.SimpleNamespace(name=s["name"], arguments=s["args"]))
                 for _, s in sorted(tc_map.items()) if s["name"]]
        message = types.SimpleNamespace(content="".join(parts), tool_calls=clean)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

    return chat


def run_one(idx: int, client, model: str):
    """Native tool-calling, ONE FORCED TOOL PER TURN (rotation):
    get_template -> semantic_search -> keyword_search -> semantic_search ->
    keyword_search -> final write (tools off). Native role:"tool" replies are
    preserved; because each turn carries exactly one forced call, replays never
    hit the multi-call streaming/replay bugs."""
    set_remote(RemoteRetrieval(ENDPOINT))
    chat_plain = make_stream_chat(client, model,
                                  session_id=f"beta-agentic-{idx}", temperature=0.2)

    def chat_forced(messages, name):
        # force a specific single tool by overriding choice per-request
        chatf = make_stream_chat(client, model,
                                 session_id=f"beta-agentic-{idx}", temperature=0.2)
        return chatf  # placeholder replaced below

    # build three chatters: forced-template, forced-search variants, free-write
    def make(forced=None, use_tools=True):
        kwargs_base = dict(session_id=f"beta-agentic-{idx}", temperature=0.2)
        if not use_tools:
            def chat_nt(messages):
                kw = dict(model=model, messages=messages, temperature=0.2,
                          max_tokens=16000, timeout=600)
                resp = client.chat.completions.create(**kw)
                return types.SimpleNamespace(choices=[types.SimpleNamespace(
                    message=resp.choices[0].message)])
            return chat_nt
        def chat_f(messages):
            kw = dict(model=model, messages=messages, temperature=0.2,
                      max_tokens=16000, timeout=600, stream=True,
                      tools=TOOL_SCHEMAS)
            if forced:
                kw["tool_choice"] = {"type": "function",
                                     "function": {"name": forced}}
            else:
                kw["tool_choice"] = "auto"
            Path(f"{HERE}/beta_experiment/last_request_{forced or 'auto'}.json") \
                .write_text(json.dumps(
                    {**{k: v for k, v in kw.items() if k != "messages"},
                     "messages": kw["messages"]},
                    ensure_ascii=False, default=str), encoding="utf-8")
            parts, tc_map = [], {}
            import time as _tt
            _t0 = _tt.time(); _last = _t0; _n = 0
            print(f"[stream] open", flush=True)
            for chunk in client.chat.completions.create(**kw):
                _now = _tt.time()
                if _now - _last > 90:
                    print(f"[stream][warn] gap {_now-_last:.0f}s", flush=True)
                _last = _now; _n += 1
                if _n == 1:
                    print(f"[stream] first chunk {(_now-_t0):.1f}s", flush=True)
                if not chunk.choices:
                    continue
                d = chunk.choices[0].delta
                # INDEX-keyed accumulation (litellm #11407: ids missing after
                # first chunk; #20480: trailing duplicate-summary chunk).
                # Ids are captured from whichever chunk carries them first.
                if d is None:
                    continue
                if getattr(d, "content", None):
                    parts.append(d.content)
                for tc in getattr(d, "tool_calls", None) or []:
                    idx = getattr(tc, "index", 0) or 0
                    slot = tc_map.setdefault(idx, {"id": "", "name": "", "args": ""})
                    if tc.id and not slot["id"]:
                        slot["id"] = tc.id          # first id wins, ignore rest
                    fname = getattr(tc.function, "name", None)
                    if fname and not slot["name"]:
                        slot["name"] = fname
                    cand = getattr(tc.function, "arguments", None)
                    if cand:
                        # duplicate-summary guard (#20480): final chunk repeats
                        # the WHOLE args string - skip exact repeats
                        if cand != slot["args"]:
                            slot["args"] += cand
            print(f"[stream] done chunks={_n} {(_tt.time()-_t0):.1f}s "
                  f"content={len(parts)}ch calls={len(tc_map)}", flush=True)
            message = types.SimpleNamespace(
                content="".join(parts),
                tool_calls=[types.SimpleNamespace(
                    id=s["id"] or f"call_{k}", type="function",
                    function=types.SimpleNamespace(name=s["name"], arguments=s["args"]))
                    for k, s in sorted(tc_map.items())])
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])
        return chat_f

    schedule = ["get_template", "semantic_search", "keyword_search",
                "semantic_search", "keyword_search"]

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content":
         "Rozszerz wzór umowy zlecenia o 3-6 nowych postanowień dla różnych "
         "sytuacji klientów. Wykonuj POJEDYNCZE wywołania narzędzi z planu "
         "(wzór -> wyszukiwania), a na końcu zwróć finalny JSON edycji."},
    ]
    trace, edits = [], []
    read_set = set()

    for step_name in schedule:
        chat_f = make(forced=step_name)
        # NO user message here: assistant(tool_calls) must be followed by the
        # matching role:"tool" reply with NOTHING in between (OpenAI spec;
        # wedging a user turn caused provider 400). Steering rides in the
        # initial task message + tool-result tails.
        print(f"[run{idx}] forced {step_name}", flush=True)
        try:
            resp = chat_f(messages)
        except Exception:
            print(f"[run{idx}] FAILED - payload kept: last_request_t*.json", flush=True)
            raise
        msg = resp.choices[0].message
        msg.tool_calls = _sanitize_tool_calls(msg.tool_calls or [],
                                              session_key=f"{idx}")
        tc = msg.tool_calls[0] if msg.tool_calls else None
        # serialize assistant turn natively (with its tool_call)
        messages.append({"role": "assistant",
                         "content": (msg.content or "").strip() or None,
                         "tool_calls": [{"id": getattr(tc, "id", "call_x"),
                                         "type": "function",
                                         "function": {"name": tc.function.name,
                                                      "arguments": tc.function.arguments}}
                                        for tc in ([tc] if tc else [])
                                        ] } )
        if not tc:
            trace.append({"step": step_name, "note": "no tool call emitted"})
            # strip nothing; next forced turn continues
            continue
        try:
            args = json.loads(tc.function.arguments or "{}")
        except Exception:
            args = {}
        if tc.function.name == "get_template" and not args.get("doc_type"):
            args["doc_type"] = "umowa_zlecenia"
        if tc.function.name == "chunk_read" and "exclude_ids" not in args:
            args["exclude_ids"] = sorted(read_set)
        print(f"[run{idx}] exec {tc.function.name} args={str(args)[:90]}", flush=True)
        result = None
        for attempt in (1, 2):
            try:
                import time as _t
                if attempt == 2:
                    _t.sleep(3)
                result = execute_tool(None, tc.function.name, args)
                break
            except Exception as exc:
                if attempt == 2:
                    names = [t["function"]["name"] for t in TOOL_SCHEMAS]
                    result = [{"error": f"{type(exc).__name__}: {exc}",
                               "hint": f"available tools: {names}"}]
        if tc.function.name == "chunk_read":
            for rr in (result if isinstance(result, list) else []):
                cid = rr.get("chunk_id") if isinstance(rr, dict) else None
                if cid and isinstance(rr, dict) and not rr.get("already_read"):
                    read_set.add(str(cid))
        if isinstance(result, list) and result and isinstance(result[0], dict) \
                and "error" in result[0]:
            names = [t["function"]["name"] for t in TOOL_SCHEMAS]
            result[0]["hint"] = f"available tools: {names}"
        trace.append({"step": step_name, "tool": tc.function.name,
                      "args_head": str(args)[:80], "items": len(result)})
        # native tool reply (single id matches the single call)
        # RAW TEXT like opencode (no JSON envelope): join items as readable
        # blocks; content is opaque to the API anyway
        if isinstance(result, list):
            body = "\n\n".join(
                (r.get("text") if isinstance(r, dict) and r.get("text")
                 else json.dumps(r, ensure_ascii=False))
                for r in result)
        else:
            body = str(result)
        if len(body) > 60000:
            print(f"[run{idx}] tool result truncated {len(body)}->60000", flush=True)
            body = body[:60000]
        messages.append({"role": "tool", "tool_call_id": getattr(tc, "id", "") or "call_x",
                         "content": body +
                          "\n\n[Kontynuuj: następne narzędzie z planu "
                          "(semantic_search/keyword_search) albo, gdy masz "
                          "dość materiału, zakończ — zarządamy finalny JSON.]"
                          + ("\n[OSTATECZNY KROK. " + FORCED_LAST + "]"
                             if step_name == schedule[-1] else "")})

    # FINAL WRITE: history already ends with the last tool reply whose tail
    # carries FORCED_LAST - provider continues straight into the JSON.
    chat_w = make(use_tools=False)
    resp = chat_w(messages)
    raw_write = resp.choices[0].message.content or ""
    print(f"[run{idx}] WRITE: {len(raw_write)}ch | head: {raw_write[:200]}", flush=True)
    parsed = extract_json(raw_write)
    if not parsed:
        print(f"[run{idx}] WRITE: extract_json FAILED on {len(raw_write)}ch", flush=True)
        print(f"[run{idx}] WRITE tail: {raw_write[-300:]}", flush=True)
    edits = (parsed or {}).get("edits", [])
    trace.append({"step": "write", "edits": len(edits), "raw_len": len(raw_write)})
    print(f"[run{idx}] DONE edits={len(edits)}", flush=True)
    return {"edits": edits, "trace": trace}


def main() -> int:
    env = load_env(os.path.join(HERE, "..", "..", "..", ".env"))
    from openai import OpenAI
    client = OpenAI(base_url=env["TEACHER_BASE_URL"], api_key=env["TEACHER_API_KEY"])
    model = env["TEACHER_MODEL"]

    results, errors = {}, {}

    def work(i):
        try:
            results[f"run{i}"] = run_one(i, client, model)
        except Exception as e:
            errors[f"run{i}"] = f"{type(e).__name__}: {e}"

    workers = int(os.environ.get("BETA_WORKERS", "3"))
    run_ids = list(range(1, int(os.environ.get("BETA_RUNS", "4"))))[:max(workers,1)] \
        if False else [i for i in (1, 2, 3)]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(work, run_ids[:workers] if workers < 3 else run_ids))

    out_dir = HERE / "beta_experiment"
    for rid, data in results.items():
        (out_dir / f"agentic_{rid}.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

    lines = []
    bank_refs = None
    for rid, data in sorted(results.items()):
        eds = data.get("edits", [])
        ops = {}
        for e in eds:
            ops[e.get("op")] = ops.get(e.get("op"), 0) + 1
        n_calls = sum(len(s.get("calls", [])) for s in data.get("trace", []))
        lines.append(f"{rid}: {len(eds)} edits {ops} | tool_calls={n_calls} "
                     f"| iters={len(data.get('trace', []))}")
        for e in eds:
            lines.append(f"   [{e.get('op')}] {str(e.get('target'))[:60]} "
                         f"cite={str(e.get('cite_ref'))[:44]}")
    if errors:
        lines.append("ERRORS: " + json.dumps(errors))
    digest = "\n".join(lines)
    (out_dir / "compare_agentic.txt").write_text(digest, encoding="utf-8")
    print(digest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
