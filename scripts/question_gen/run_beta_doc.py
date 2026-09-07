"""PHASE BETA v2 - TRUE agentic template editing (user's original vision).

Each independent run: AI is told to FIRST read the template via get_template
tool, THEN pull whatever grounding material IT decides via semantic_search /
keyword_search / chunk_read, and only then emit structured add/remove/modify
edits. Full tool-call transcript recorded per run so we can SEE what it chose
to retrieve and what it added/removed.

Retrieval goes through the RAG service (set_remote -> RETRIEVAL_ENDPOINT,
default http://127.0.0.1:10100; see scripts/rag/server.py). Streaming per agent-1's
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
sys.path.insert(0, str(HERE / ".." / ".." / "src"))

from openai import OpenAI                                    # noqa: E402
from legal_drafter.retrieval.agentic_tools import (          # noqa: E402
    TOOL_SCHEMAS, execute_tool, set_remote,
)
from legal_drafter.retrieval.remote import RemoteRetrieval   # noqa: E402

ENDPOINT = "http://127.0.0.1:10100"
MAX_ITER = 6

def system_research(doc_type: str) -> str:
    return (
        "You are an experienced Polish legal drafter researching material to "
        f"create a realistic VARIATION of a contract template (doc_type='{doc_type}'). "
        "Each variation represents what a DIFFERENT client would need.\n\n"
        "RETRIEVAL TOOLS:\n"
        "- get_template(doc_type, query) — read the base template (query is optional detail)\n"
        "- semantic_search(query) — semantic search over the corpus\n"
        "- keyword_search(keywords) — keyword search\n"
        "- chunk_read(chunk_ids) — full text of specific chunks\n\n"
        "Read the base template, then search for provisions relevant to the "
        "scenario direction provided. Ground everything in real material."
    )

def system_draft(doc_type: str) -> str:
    return (
        "You are an experienced Polish legal drafter. Your ONLY task is to write "
        f"the COMPLETE modified {doc_type} contract in Polish.\n\n"
        "You will receive the RESEARCH HISTORY (base template + retrieved legal "
        "provisions). Rules:\n"
        "- Consider ONLY the RELEVANT chunks from the history; ignore noise.\n"
        "- You may ADD, REMOVE, or REPLACE clauses in the base template to apply "
        "the variation described in the initial goal — but every change must be "
        "grounded in the retrieved provisions from the history. Do NOT invent "
        "provisions, facts, or figures not present in the history.\n"
        "- Start from the base template and use the retrieved provisions as support.\n"
        "- Write a full, ready-to-use contract with all standard sections."
    )

SYSTEM_QUESTIONS = (
    "You write layperson questions in Polish that a client would ask to obtain "
    "the given contract. Rules:\n"
    "- Two questions: 'prosta' (1-3 short sentences, pure situation + need) and "
    "'szczegolowa' (2-5 sentences with concrete details).\n"
    "- Questions must NOT contain legal citations, article numbers, or lawyer "
    "voice. They should sound like a real person describing their situation.\n"
    "- Base them on what the contract actually covers."
)

FORCED_DRAFT = (
    "Return ONLY strict JSON (no markdown):\n"
    '{"summary": "<one-line: what client scenario this variation represents>",\n'
    ' "contract": "<the COMPLETE modified contract in Polish, all sections>"}'
)

FORCED_QUESTIONS = (
    "Return ONLY strict JSON (no markdown):\n"
    '{"questions": [\n'
    '  {"variant": "prosta", "text": "<1-3 sentences in Polish>"},\n'
    '  {"variant": "szczegolowa", "text": "<2-5 sentences in Polish>"}\n'
    ' ]}'
)


def load_env(path=".env"):
    env = dict(os.environ)          # real environment takes precedence
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env.setdefault(k.strip(), v.strip())
    return env


def _escape_json_string_values(s):
    """Escape raw control chars (newline, tab, CR) inside JSON string values.

    The base 2B model emits raw line breaks inside string values (e.g. the
    ``contract`` field), which makes ``json.loads`` fail. This walks the text
    and escapes control chars only inside quoted string values, leaving the
    structural whitespace between keys untouched."""
    out = []
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        if c == '"':
            j = i + 1
            while j < n:
                c2 = s[j]
                if c2 == '\\':
                    j += 2
                    continue
                if c2 == '"':
                    break
                j += 1
            val = s[i:j + 1]
            val = val.replace('\r', '\\r').replace('\n', '\\n').replace('\t', '\\t')
            out.append(val)
            i = j + 1
        else:
            out.append(c)
            i += 1
    return ''.join(out)


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
        pass
    # Retry after escaping raw control chars inside string values.
    try:
        return json.loads(_escape_json_string_values(m.group(0)))
    except Exception:
        return None


def _parse_text_tag_args(raw: str) -> str:
    raw_stripped = (raw or "").strip()
    try:
        return json.dumps(json.loads(raw_stripped), ensure_ascii=False)
    except Exception:
        pass
    keys = re.findall(r"<arg_key[^>]*>(.*?)</arg_key", raw_stripped)
    vals = re.findall(r"<arg_value[^>]*>(.*?)</arg_value", raw_stripped)
    if keys and vals and len(keys) == len(vals):
        obj = {}
        for k, v in zip(keys, vals):
            k, v = k.strip(), v.strip()
            try:
                v = json.loads(v)
            except Exception:
                pass
            obj[k] = v
        return json.dumps(obj, ensure_ascii=False)
    try:
        obj, _ = json.JSONDecoder().raw_decode(raw_stripped)
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        pass
    i = raw_stripped.find("{")
    if i >= 0:
        try:
            obj, _ = json.JSONDecoder().raw_decode(raw_stripped[i:])
            return json.dumps(obj, ensure_ascii=False)
        except Exception:
            pass
    return "{}"


def _sanitize_tool_calls(calls, session_key: str):
    out, seen = [], set()
    for k, tc in enumerate(calls):
        name = tc.function.name or f"tool_{k}"
        if not name or name in ("ExaWeb-web_search_exa", "web_search_exa",
                                "web_fetch_exa", "ExaWeb-web_fetch_exa"):
            continue
        args = _parse_text_tag_args(tc.function.arguments)
        cid = tc.id or ""
        if not cid or cid in seen:
            cid = f"call_{session_key}_{k}"
        seen.add(cid)
        out.append(types.SimpleNamespace(
            id=cid, type="function",
            function=types.SimpleNamespace(name=name, arguments=args)))
    return out


def is_successful_result(result):
    """A tool call counts as successful only if it returned real content."""
    if not isinstance(result, list) or not result:
        return False
    for r in result:
        if isinstance(r, dict) and "error" not in r and (
            r.get("text") or r.get("chunk_id")
            or r.get("kind") == "structural_template"):
            return True
    return False


class CallCache:
    """Detects repeated identical tool calls (loop guard)."""

    def __init__(self, max_repeats: int = 3):
        self.seen = {}
        self.max_repeats = max_repeats

    def over_limit(self, tool: str, args: dict) -> bool:
        key = (tool, json.dumps(args, sort_keys=True, ensure_ascii=False))
        self.seen[key] = self.seen.get(key, 0) + 1
        return self.seen[key] > self.max_repeats


class NotApplicable:
    """Thread-safe record of scenarios that have no supporting RAG data.

    A scenario is added here when >=3 of its tool calls return empty (no
    relevant material). Workers skip anything already in this set, so a dead
    scenario is tried once and then never wastes another run.
    """

    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.seen = set()
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    self.seen.add(line)

    def record(self, hint: str):
        with self.lock:
            if hint and hint not in self.seen:
                self.seen.add(hint)
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(hint + "\n")

    def __contains__(self, hint: str) -> bool:
        return hint in self.seen


def _stream_turn(client, model, messages, tools=None, temperature=0.2, extra_body=None):
    """One streaming turn. Returns SimpleNamespace with .content, .reasoning, .tool_calls."""
    # Cap completion length; vLLM's --max-model-len is 80000 so we allow the
    # full ~65k contract budget requested for eval.
    _mt = min(int(os.environ.get("QA_MAX_TOKENS", "65536")), 70000)
    kwargs = dict(model=model, messages=messages, temperature=temperature,
                  max_tokens=_mt, timeout=600, stream=True)
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    if extra_body:
        kwargs["extra_body"] = extra_body
    kwargs["stream_options"] = {"include_usage": True}
    parts, reasoning_parts, tc_map = [], [], {}
    import time as _tt
    _t0 = _tt.time()
    usage = None
    for chunk in client.chat.completions.create(**kwargs):
        if getattr(chunk, "usage", None) is not None:
            usage = chunk.usage
        if not chunk.choices:
            continue
        d = chunk.choices[0].delta
        if d is None:
            continue
        if getattr(d, "content", None):
            parts.append(d.content)
        r = getattr(d, "reasoning", None) or getattr(d, "reasoning_content", None)
        if r:
            reasoning_parts.append(r)
        for tc in getattr(d, "tool_calls", None) or []:
            i = getattr(tc, "index", 0) or 0
            slot = tc_map.setdefault(i, {"id": "", "name": "", "args": ""})
            if tc.id and not slot["id"]:
                slot["id"] = tc.id
            fname = getattr(tc.function, "name", None)
            if fname and not slot["name"]:
                slot["name"] = fname
            cand = getattr(tc.function, "arguments", None)
            if cand and cand != slot["args"]:
                slot["args"] += cand
    content = "".join(parts)
    reasoning = "".join(reasoning_parts)
    tool_calls = [types.SimpleNamespace(
        id=s["id"] or f"call_{k}", type="function",
        function=types.SimpleNamespace(name=s["name"], arguments=s["args"]))
        for k, s in sorted(tc_map.items())]
    _dt = _tt.time() - _t0
    put = (usage.prompt_tokens if usage else 0)
    cut = (usage.completion_tokens if usage else 0)
    print(f"[stream] {_dt:.1f}s content={len(content)}ch "
          f"reasoning={len(reasoning)}ch calls={len(tool_calls)} "
          f"tokens(p={put} c={cut})", flush=True)
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(
            content=content, reasoning=reasoning, tool_calls=tool_calls))],
        usage=usage, _elapsed=_dt)


COVERED_FILE = None


def load_covered():
    if COVERED_FILE and COVERED_FILE.exists():
        return [l.strip() for l in COVERED_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]
    return []


def save_covered(summaries):
    if COVERED_FILE:
        with open(COVERED_FILE, "a", encoding="utf-8") as f:
            for s in summaries:
                f.write(s + "\n")


def run_one(idx: int, client, model: str, endpoint: str, doc_type: str,
            hint: str, na_state: NotApplicable):
    set_remote(RemoteRetrieval(endpoint))
    covered = load_covered()
    if not hint:
        hint = ""

    # --- PHASE 1: RESEARCH (streaming + tools) ---
    system = system_research(doc_type)
    if covered:
        system += "\n\nPreviously created variations (do NOT repeat):\n"
        for c in covered:
            system += f"- {c}\n"
    system += f"\nScenario direction: {hint}."

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content":
         "Research material for a variation of this contract. "
         "Read the base template, then search for provisions relevant to the "
         "scenario direction. You have 4 tool turns."},
    ]
    trace = []
    read_set = set()
    cache = CallCache(max_repeats=3)
    success = 0
    empty = 0
    MAX_ITER = 10  # hard safety net; we break at 5 successful calls

    for it in range(MAX_ITER):
        print(f"[run{idx}] P1 turn {it} (success={success})", flush=True)
        resp = _stream_turn(client, model, messages, tools=TOOL_SCHEMAS)
        msg = resp.choices[0].message
        msg.tool_calls = _sanitize_tool_calls(msg.tool_calls or [], str(idx))
        if not msg.tool_calls:
            trace.append({"p1_turn": it, "note": "no tool calls, stopping research"})
            break
        messages.append({"role": "assistant", "content": msg.content or None,
                         "reasoning": msg.reasoning or "",
                         "tool_calls": [{"id": tc.id, "type": "function",
                                         "function": {"name": tc.function.name,
                                                      "arguments": tc.function.arguments}}
                                        for tc in msg.tool_calls]})
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except Exception:
                args = _parse_text_tag_args(tc.function.arguments)
                args = json.loads(args) if args else {}
            if not isinstance(args, dict):
                args = {}
            if tc.function.name == "get_template" and not args.get("doc_type"):
                args["doc_type"] = doc_type
            if tc.function.name == "chunk_read" and "exclude_ids" not in args:
                args["exclude_ids"] = sorted(read_set)
            print(f"[run{idx}]   {tc.function.name} {str(args)[:80]}", flush=True)
            if cache.over_limit(tc.function.name, args):
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": json.dumps(
                                     [{"warning": "duplicate call suppressed; "
                                                 "try a different approach"}])})
                continue
            try:
                result = execute_tool(None, tc.function.name, args)
            except Exception as exc:
                result = [{"error": str(exc)}]
            if tc.function.name == "chunk_read":
                for rr in (result if isinstance(result, list) else []):
                    cid = rr.get("chunk_id") if isinstance(rr, dict) else None
                    if cid:
                        read_set.add(str(cid))
            blob = json.dumps(result, ensure_ascii=False)
            messages.append({"role": "tool", "tool_call_id": tc.id,
                             "content": blob[:60000]})
            if is_successful_result(result):
                success += 1
            elif any(isinstance(r, dict) and "error" in r
                     for r in (result if isinstance(result, list) else [])):
                # transient/error response — don't count toward "no data" skip
                pass
            else:
                empty += 1
                if empty >= 3:
                    print(f"[run{idx}] scenario has no relevant RAG data "
                          f"(>=3 empty tool responses); marking not-applicable: "
                          f"{hint[:60]}", flush=True)
                    na_state.record(hint)
                    return ("SKIP", hint)
        if success >= 5:
            print(f"[run{idx}] reached 5 successful tool calls, stopping research",
                  flush=True)
            break

    # --- PHASE 2: DRAFTING BRIEF (streaming, NO tools) ---
    print(f"[run{idx}] P2 drafting brief", flush=True)
    transcript = []
    for m in messages:
        role = m["role"]
        if role == "system":
            continue
        if role == "user":
            transcript.append(f"USER TASK: {m.get('content','')}")
        elif role == "assistant":
            if m.get("content"):
                transcript.append(f"DRAFTER: {m['content']}")
            for tc in (m.get("tool_calls") or []):
                fn = tc["function"]["name"]
                args = tc["function"]["arguments"]
                transcript.append(f"DRAFTER CALLED {fn}({args})")
        elif role == "tool":
            try:
                parsed = json.loads(m.get("content") or "[]")
            except Exception:
                parsed = None
            if isinstance(parsed, list):
                # keep full template text; cap other chunk text at 3000 chars
                items = []
                for r in parsed:
                    if isinstance(r, dict):
                        r = dict(r)
                        if "text" in r and len(r["text"]) > 3000 and r.get("kind") != "structural_template":
                            r["text"] = r["text"][:3000] + " …[truncated]"
                        items.append(r)
                    else:
                        items.append(r)
                blob = json.dumps(items, ensure_ascii=False)
            else:
                blob = (m.get("content") or "")[:6000]
            transcript.append(f"RESULT: {blob}")
    transcript_text = "\\n".join(transcript)

    p2_messages = [
        {"role": "system", "content": system_draft(doc_type)},
        {"role": "user", "content": (
            f"INITIAL GOAL: making a {hint} {doc_type} contract, applying the "
            f"extra applicable provisions found in the research.\n\n"
            "RESEARCH HISTORY (use ONLY this material):\n" + transcript_text +
            "\n\n" + FORCED_DRAFT
        )},
    ]
    p2_resp = _stream_turn(client, model, p2_messages)
    p2_raw = p2_resp.choices[0].message.content or ""
    print(f"[run{idx}] P2 draft: {len(p2_raw)}ch", flush=True)
    p2_parsed = extract_json(p2_raw)
    if not p2_parsed:
        print(f"[run{idx}] P2 extract_json FAILED", flush=True)
    summary = (p2_parsed or {}).get("summary", "")
    contract = (p2_parsed or {}).get("contract", "")

    # --- PHASE 3: QUESTIONS from contract (streaming, NO tools) ---
    print(f"[run{idx}] P3 questions", flush=True)
    p3_system = SYSTEM_QUESTIONS + (
        f"\n\nINITIAL GOAL: {hint}\n\nCONTRACT:\n" + (contract or "")[:45000]
    )
    p3_messages = [
        {"role": "system", "content": p3_system},
        {"role": "user", "content": FORCED_QUESTIONS},
    ]
    p3_resp = _stream_turn(client, model, p3_messages)
    raw = p3_resp.choices[0].message.content or ""
    print(f"[run{idx}] P3 output: {len(raw)}ch", flush=True)
    parsed = extract_json(raw)

    variation = {
        "run": idx,
        "hint": hint,
        "summary": summary,
        "contract": contract,
        "questions": (parsed or {}).get("questions", []),
        "research_summary": transcript_text[:2000],
    }
    # Distillation-ready trajectory: full Phase-1 research trace (with reasoning)
    # + the drafted contract. Phase-3 questions are the input framings.
    trajectory = {
        "run": idx,
        "hint": hint,
        "scenario": hint,
        "tools": TOOL_SCHEMAS,
        "messages": [
            {"role": "system", "content": system_research(doc_type)},
            {"role": "user", "content":
             "Research material for a variation of this contract. "
             "Read the base template, then search for provisions relevant to the "
             "scenario direction. You have 4 tool turns."},
            *[{"role": m["role"],
               "content": m.get("content"),
               **({"reasoning": m["reasoning"]} if m.get("reasoning") else {}),
               **({"tool_calls": m["tool_calls"]} if m.get("tool_calls") else {}),
               **({"tool_call_id": m["tool_call_id"], "name": m.get("name")}
                  if m["role"] == "tool" else {})}
              for m in messages if m["role"] != "system"],
            {"role": "assistant", "content": contract or "",
             "note": "Phase-2 drafted contract (final answer)"},
        ],
        "contract": contract,
        "questions": (parsed or {}).get("questions", []),
    }
    if not variation["contract"]:
        print(f"[run{idx}] WARN: empty contract", flush=True)

    if variation["summary"]:
        save_covered([variation["summary"]])
    print(f"[run{idx}] DONE: contract={len(variation['contract'])}ch "
          f"questions={len(variation['questions'])}", flush=True)
    return variation, trajectory


def main() -> int:
    global COVERED_FILE
    env = load_env(os.path.join(HERE, "..", "..", ".env"))

    # --- parametric doc-type config ---
    doc_type = os.environ.get("DOC_TYPE", "umowa_zlecenia")
    out_dir_raw = os.environ.get("OUT_DIR")
    if out_dir_raw:
        out_dir = Path(out_dir_raw)
    else:
        repo_root = HERE.parent.parent          # outputs/beta_pipeline -> repo root
        out_dir = repo_root / "outputs" / "anchors" / doc_type
    out_dir.mkdir(parents=True, exist_ok=True)
    COVERED_FILE = out_dir / "covered.txt"

    # scenarios: explicit SCENARIOS file, else <out_dir>/hints.txt
    hints = []
    scenarios_path = os.environ.get("SCENARIOS")
    if scenarios_path and Path(scenarios_path).exists():
        hints = [l.strip() for l in Path(scenarios_path).read_text(encoding="utf-8").splitlines() if l.strip()]
    elif (out_dir / "hints.txt").exists():
        hints = [l.strip() for l in (out_dir / "hints.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    if hints:
        print(f"[beta] loaded {len(hints)} scenarios", flush=True)
    else:
        print("[beta] WARNING: no scenarios provided (SCENARIOS or hints.txt); "
              "runs will have no scenario direction", flush=True)

    from openai import OpenAI
    client = OpenAI(base_url=env["TEACHER_BASE_URL"],
                    api_key=env["CHECKER_API_KEY"] or env["TEACHER_API_KEY"])
    model = env["CHECKER_MODEL"]
    endpoint = os.environ.get(
        "RETRIEVAL_ENDPOINT",
        "http://127.0.0.1:10100")
    print(f"[beta] doc_type={doc_type} model={model} out_dir={out_dir}", flush=True)

    # Warm up the retrieval endpoint so the first tool call isn't a cold start
    try:
        warm = RemoteRetrieval(endpoint)
        print("[beta] warming up retrieval endpoint...", flush=True)
        warm.get_template(doc_type)
        print("[beta] endpoint warm.", flush=True)
    except Exception as e:
        print(f"[beta] warmup warning: {e}", flush=True)

    n_runs = int(os.environ.get("BETA_RUNS", "3"))
    n_workers = int(os.environ.get("BETA_WORKERS", "1"))

    na_state = NotApplicable(out_dir / "not_applicable.txt")
    todo = [h for h in hints if h not in na_state] or [""]
    print(f"[beta] {len(todo)} scenario(s) queued "
          f"({len(hints) - len(todo)} already not-applicable)", flush=True)

    todo_lock = threading.Lock()
    cursor = [0]
    completed = [0]
    completed_lock = threading.Lock()

    def worker(wid):
        while True:
            with todo_lock:
                if cursor[0] >= len(todo):
                    return
                with completed_lock:
                    if completed[0] >= n_runs:
                        return
                run_id = cursor[0] + 1
                hint = todo[cursor[0]]
                cursor[0] += 1
            try:
                res = run_one(run_id, client, model, endpoint, doc_type, hint, na_state)
            except Exception as e:
                print(f"[worker{wid}] ERROR on scenario '{hint[:50]}': {e!r}",
                      flush=True)
                continue
            if isinstance(res, tuple) and res and res[0] == "SKIP":
                continue  # recorded as not-applicable; pull next scenario
            variation, trajectory = res
            with completed_lock:
                completed[0] += 1
                n = completed[0]
            out_path = out_dir / f"variation_{n}.json"
            out_path.write_text(json.dumps(variation, ensure_ascii=False, indent=1),
                                encoding="utf-8")
            traj_path = out_dir / f"trajectory_{n}.json"
            traj_path.write_text(json.dumps(trajectory, ensure_ascii=False, indent=1),
                                 encoding="utf-8")
            print(f"[save] variation_{n}.json + trajectory_{n}.json", flush=True)

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        list(ex.map(worker, range(n_workers)))

    lines = []
    for f in sorted(out_dir.glob("variation_*.json")):
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        lines.append(f"{f.name}: contract={len(d.get('contract',''))}ch "
                     f"questions={len(d.get('questions',[]))} | "
                     f"{d.get('summary','')[:60]}")
    lines.append(f"not_applicable scenarios: {len(na_state.seen)} "
                 f"(see {na_state.path.name})")
    digest = "\n".join(lines)
    (out_dir / "compare_agentic.txt").write_text(digest, encoding="utf-8")
    print(f"[beta] DONE: completed={completed[0]} "
          f"not_applicable={len(na_state.seen)}", flush=True)
    print(digest, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
