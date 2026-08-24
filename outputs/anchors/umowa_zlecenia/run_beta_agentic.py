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

SYSTEM_RESEARCH = (
    "You are an experienced Polish legal drafter researching material to "
    "create a realistic VARIATION of a mandate contract (umowa zlecenia) "
    "template. Each variation represents what a DIFFERENT client would need.\n\n"
    "RETRIEVAL TOOLS:\n"
    "- get_template(doc_type, query) — read the base template (query is optional detail)\n"
    "- semantic_search(query) — semantic search over the corpus\n"
    "- keyword_search(keywords) — keyword search\n"
    "- chunk_read(chunk_ids) — full text of specific chunks\n\n"
    "Read the base template, then search for provisions relevant to the "
    "scenario direction provided. Ground everything in real material."
)

SYSTEM_DRAFT = (
    "You are an experienced Polish legal drafter. Your ONLY task is to write "
    "the COMPLETE modified mandate contract (umowa zlecenia) in Polish.\n\n"
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


def _stream_turn(client, model, messages, tools=None, temperature=0.2):
    """One streaming turn. Returns SimpleNamespace with .content, .reasoning, .tool_calls."""
    kwargs = dict(model=model, messages=messages, temperature=temperature,
                  max_tokens=16000, timeout=600, stream=True)
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    parts, reasoning_parts, tc_map = [], [], {}
    import time as _tt
    _t0 = _tt.time()
    for chunk in client.chat.completions.create(**kwargs):
        if not chunk.choices:
            continue
        d = chunk.choices[0].delta
        if d is None:
            continue
        if getattr(d, "content", None):
            parts.append(d.content)
        if getattr(d, "reasoning", None):
            reasoning_parts.append(d.reasoning)
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
    print(f"[stream] {(_tt.time()-_t0):.1f}s content={len(content)}ch "
          f"reasoning={len(reasoning)}ch calls={len(tool_calls)}", flush=True)
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(
            content=content, reasoning=reasoning, tool_calls=tool_calls))])


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


def run_one(idx: int, client, model: str, endpoint: str):
    set_remote(RemoteRetrieval(endpoint))
    covered = load_covered()
    hints = []
    hints_path = HERE / "beta_experiment" / "hints.txt"
    if hints_path.exists():
        hints = [l.strip() for l in hints_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    hint = hints[(idx - 1) % len(hints)] if hints else ""

    # --- PHASE 1: RESEARCH (streaming + tools) ---
    system = SYSTEM_RESEARCH
    if covered:
        system += "\n\nPreviously created variations (do NOT repeat):\n"
        for c in covered:
            system += f"- {c}\n"
    system += f"\nScenario direction: {hint}."

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content":
         "Research material for a variation of this mandate contract. "
         "Read the template, then search for provisions relevant to the "
         "scenario direction. You have 4 tool turns."},
    ]
    trace = []
    read_set = set()
    cache = CallCache(max_repeats=3)
    success = 0
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
                args["doc_type"] = "umowa_zlecenia"
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
        {"role": "system", "content": SYSTEM_DRAFT},
        {"role": "user", "content": (
            f"INITIAL GOAL: making a {hint} mandate contract, applying the "
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
            {"role": "system", "content": SYSTEM_RESEARCH},
            {"role": "user", "content":
             "Research material for a variation of this mandate contract. "
             "Read the template, then search for provisions relevant to the "
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
    COVERED_FILE = HERE / "beta_experiment" / "covered.txt"
    env = load_env(os.path.join(HERE, "..", "..", "..", ".env"))
    from openai import OpenAI
    client = OpenAI(base_url=env["TEACHER_BASE_URL"],
                    api_key=env["CHECKER_API_KEY"] or env["TEACHER_API_KEY"])
    model = env["CHECKER_MODEL"]
    endpoint = os.environ.get(
        "RETRIEVAL_ENDPOINT",
        "https://nikkybrekas--legal-drafter-retrieval-retrievalservicegpu-app.modal.run")
    print(f"[beta] model={model} endpoint={endpoint[:60]}...", flush=True)

    # Warm up the Modal retrieval endpoint so the first tool call isn't a cold start
    try:
        warm = RemoteRetrieval(endpoint)
        print("[beta] warming up retrieval endpoint...", flush=True)
        warm.get_template("umowa_zlecenia")
        print("[beta] endpoint warm.", flush=True)
    except Exception as e:
        print(f"[beta] warmup warning: {e}", flush=True)

    out_dir = HERE / "beta_experiment"
    n_runs = int(os.environ.get("BETA_RUNS", "3"))
    workers = int(os.environ.get("BETA_WORKERS", "1"))
    run_ids = list(range(1, n_runs + 1))

    results, errors = {}, {}

    def work(i):
        try:
            variation, trajectory = run_one(i, client, model, endpoint)
            results[f"run{i}"] = variation
            out_path = out_dir / f"variation_{i}.json"
            out_path.write_text(json.dumps(variation, ensure_ascii=False, indent=1),
                                encoding="utf-8")
            traj_path = out_dir / f"trajectory_{i}.json"
            traj_path.write_text(json.dumps(trajectory, ensure_ascii=False, indent=1),
                                 encoding="utf-8")
            print(f"[save] variation_{i}.json + trajectory_{i}.json", flush=True)
        except Exception as e:
            errors[f"run{i}"] = f"{type(e).__name__}: {e}"
            print(f"[run{i}] ERROR: {e}", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(work, run_ids))

    lines = []
    for rid, data in sorted(results.items()):
        lines.append(f"{rid}: contract={len(data.get('contract',''))}ch "
                     f"questions={len(data.get('questions',[]))} | "
                     f"{data.get('summary','')[:60]}")
    if errors:
        lines.append(f"ERRORS: {json.dumps(errors)}")
    digest = "\n".join(lines)
    (out_dir / "compare_agentic.txt").write_text(digest, encoding="utf-8")
    print(digest, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
