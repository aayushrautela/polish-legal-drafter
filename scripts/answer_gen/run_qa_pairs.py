"""Question->Answer distillation-pair run (parallel, batched).

For each layperson question (prosta / szczegolowa), run ONE coherent agentic
episode: the question is the user input, the model researches with RAG tools,
then drafts the full contract. Reasoning tokens are captured on every assistant
turn. The saved artifact is the distillation pair:

    input  = the client question
    target = research trajectory (reasoning + tool calls + results) + contract

This is where reasoning must come from -- NOT from the 3-phase generator.

Parallelization: a ThreadPoolExecutor runs many questions concurrently. Each
worker streams independently against the shared OpenAI client; results are
appended under a lock to JSONL files and can be resumed (ids already present
in qa_pairs.jsonl are skipped).

Clean pair stored per line in qa_pairs.jsonl:
    {id, doc_type, variant, source, question, reasoning, answer,
     n_tool_calls, n_successful, empty}

The full episode (with tool calls + results) is kept in qa_pairs_full.jsonl
for debugging / later re-distillation.
"""
import json
import os
import re
import sys
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / ".." / "question_gen"))
sys.path.insert(0, str(HERE / ".." / ".." / "src"))

import run_beta_doc as R
from openai import OpenAI

SYSTEM_QA = (
    "You are a Polish legal drafter. Use the retrieval tools to find "
    "relevant legal provisions, then write a COMPLETE contract in Polish.\n\n"
    "Rules:\n"
    "- Use only provisions found in the retrieved chunks. Do not invent.\n"
    "- After researching, return ONLY strict JSON: {\"summary\": \"<one-line "
    "scenario>\", \"contract\": \"<full contract>\"}"
)

SYSTEM_DRAFT_QA = (
    "You are an experienced Polish legal drafter. Your ONLY task is to write "
    "the COMPLETE contract (in Polish) that fulfills the client's request, "
    "based on the research history provided.\n\n"
    "Think step by step in English before writing your response.\n\n"
    "Rules:\n"
    "- Consider ONLY the relevant chunks from the history; ignore noise.\n"
    "- Start from the base template (in the history) and use the retrieved "
    "provisions as support.\n"
    "- You may add, remove, or replace clauses in the base template to match "
    "the client's specific facts, but every change must be grounded in "
    "retrieved provisions. Do NOT invent provisions or facts not present in "
    "the history.\n"
    "- CRITICAL: Follow the template's section order and include ALL of its "
    "statutory citations (e.g. art. 734-751 KC, art. 8a ustawy o min. "
    "wynagrodzeniu). Do NOT omit standard sections (confidentiality, "
    "penalties, termination, copyright, final provisions). A contract "
    "missing citations or standard sections is legally incomplete."
)

# --- copied verbatim from run_beta_doc.py (teacher SFT generation) ---
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

FORCED_DRAFT = (
    "Think in English step by step, then return ONLY strict JSON (no markdown):\n"
    '{"summary": "<one-line: what client scenario this variation represents>",\n'
    ' "contract": "<the COMPLETE modified contract in Polish, all sections>"}'
)
# --- end copied block ---


def build_train_messages(messages, draft_reasoning, contract):
    """Convert the in-flight research conversation into the training shape
    Qwen3.5/Unsloth expect: reasoning stored as `reasoning_content`, tool
    results truncated, final contract appended as the last assistant turn."""
    out = []
    for m in messages:
        role = m["role"]
        if role == "system":
            out.append({"role": "system", "content": m.get("content", "")})
        elif role == "user":
            out.append({"role": "user", "content": m.get("content", "")})
        elif role == "assistant":
            am = {"role": "assistant", "content": m.get("content") or ""}
            if m.get("reasoning"):
                am["reasoning_content"] = m["reasoning"]
            if m.get("tool_calls"):
                am["tool_calls"] = [
                    {"id": tc["id"], "type": "function",
                     "function": {"name": tc["function"]["name"],
                                  "arguments": tc["function"]["arguments"]}}
                    for tc in m["tool_calls"]
                ]
            out.append(am)
        elif role == "tool":
            out.append({"role": "tool", "tool_call_id": m.get("tool_call_id"),
                        "content": m.get("content", "")})
    out.append({"role": "assistant", "content": contract or "",
                "reasoning_content": draft_reasoning})
    return out


def build_transcript(messages):
    out = []
    for m in messages:
        role = m["role"]
        if role == "system":
            continue
        if role == "user":
            out.append(f"USER: {m.get('content','')}")
        elif role == "assistant":
            if m.get("reasoning"):
                out.append(f"REASONING: {m['reasoning']}")
            if m.get("content"):
                out.append(f"DRAFTER: {m['content']}")
            for tc in (m.get("tool_calls") or []):
                out.append(f"DRAFTER CALLED {tc['function']['name']}({tc['function']['arguments']})")
        elif role == "tool":
            try:
                parsed = json.loads(m.get("content") or "[]")
            except Exception:
                parsed = None
            if isinstance(parsed, list):
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
            out.append(f"RESULT: {blob}")
    return "\n".join(out)


_write_lock = threading.Lock()


def run_qa(q, client, model, endpoint, tool_cap, temperature=0.2, extra_body=None):
    idx = q["id"]
    question = q["question"]
    variant = q["variant"]
    doc_type = q["doc_type"]
    source = q["source"]
    R.set_remote(R.RemoteRetrieval(endpoint))
    messages = [
        {"role": "system", "content": SYSTEM_QA},
        {"role": "user", "content": question},
    ]
    cache = R.CallCache(max_repeats=3)
    success = 0
    n_calls = 0
    reasoning_parts = []
    prompt_tokens = 0
    completion_tokens = 0
    _t_start = time.time()
    MAX_ITER = tool_cap + 6
    for it in range(MAX_ITER):
        resp = R._stream_turn(client, model, messages, tools=R.TOOL_SCHEMAS,
                              temperature=temperature, extra_body=extra_body)
        if getattr(resp, "usage", None):
            prompt_tokens += resp.usage.prompt_tokens
            completion_tokens += resp.usage.completion_tokens
        msg = resp.choices[0].message
        msg.tool_calls = R._sanitize_tool_calls(msg.tool_calls or [], str(idx))
        _r = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
        if _r:
            reasoning_parts.append(_r)
        if not msg.tool_calls:
            break
        messages.append({
            "role": "assistant",
            "content": msg.content or None,
            # Qwen3.5 expects "reasoning_content" (not "reasoning") in conversation
            # history for multi-turn tool-call flows. Using the wrong key drops
            # thinking turn-to-turn and degrades accuracy.
            "reasoning_content": msg.reasoning or "",
            "tool_calls": [{"id": tc.id, "type": "function",
                           "function": {"name": tc.function.name,
                                        "arguments": tc.function.arguments}}
                          for tc in msg.tool_calls],
        })
        for tc in msg.tool_calls:
            n_calls += 1
            try:
                args = json.loads(tc.function.arguments or "{}")
            except Exception:
                args = R._parse_text_tag_args(tc.function.arguments)
                args = json.loads(args) if args else {}
            if not isinstance(args, dict):
                args = {}
            print(f"[qa{idx}]   {tc.function.name} {str(args)[:80]}", flush=True)
            if cache.over_limit(tc.function.name, args):
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": json.dumps(
                                     [{"warning": "duplicate call suppressed; "
                                                 "try a different approach"}])})
                continue
            try:
                result = R.execute_tool(None, tc.function.name, args)
            except Exception as exc:
                result = [{"error": str(exc)}]
            blob = json.dumps(result, ensure_ascii=False)
            messages.append({"role": "tool", "tool_call_id": tc.id,
                             "content": blob[:60000]})
            if R.is_successful_result(result):
                success += 1
        if success >= 5:
            print(f"[qa{idx}] reached 5 successful tool calls, stopping research", flush=True)
            break

    # --- draft phase (no tools) -> contract (copied verbatim from run_beta_doc.py P2) ---
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
    transcript_text = "\n".join(transcript)

    draft_messages = [
        {"role": "system", "content": system_draft(doc_type)},
        {"role": "user", "content": (
            f"INITIAL GOAL: making a {variant} {doc_type} contract, applying the "
            f"extra applicable provisions found in the research.\n\n"
            "RESEARCH HISTORY (use ONLY this material):\n" + transcript_text +
            "\n\n" + FORCED_DRAFT
        )},
    ]
    dresp = R._stream_turn(client, model, draft_messages,
                          temperature=temperature, extra_body=extra_body)
    if getattr(dresp, "usage", None):
        prompt_tokens += dresp.usage.prompt_tokens
        completion_tokens += dresp.usage.completion_tokens
    _m = dresp.choices[0].message
    # With --reasoning-parser qwen3, the base 2B model may put the JSON answer
    # into .reasoning (thinking) and leave .content null. Read content first,
    # fall back to reasoning so the contract is never lost.
    draft_raw = _m.content or _m.reasoning or ""
    dparsed = R.extract_json(draft_raw)
    contract = (dparsed or {}).get("contract", "")
    draft_reasoning = _m.reasoning or ""
    if draft_reasoning:
        reasoning_parts.append(draft_reasoning)
    reasoning = "\n\n".join(p.strip() for p in reasoning_parts if p and p.strip())

    # training-ready episode (question -> research w/ reasoning+tool_calls -> contract)
    train_messages = build_train_messages(messages, draft_reasoning, contract)

    pair = {
        "id": idx,
        "doc_type": doc_type,
        "variant": variant,
        "source": source,
        "question": question,
        "reasoning_content": reasoning,
        "answer": contract,
        "draft_raw": draft_raw,
        "draft_reasoning": draft_reasoning,
        "n_tool_calls": n_calls,
        "n_successful": success,
        "empty": not contract,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "elapsed_s": round(time.time() - _t_start, 1),
    }
    full = {
        "id": idx,
        "doc_type": doc_type,
        "variant": variant,
        "source": source,
        "question": question,
        "tools": R.TOOL_SCHEMAS,
        "messages": train_messages,
        "contract": contract,
        "draft_raw": draft_raw,
        "draft_reasoning": draft_reasoning,
    }
    if not contract:
        print(f"[qa{idx}] WARN: empty contract", flush=True)
    print(f"[qa{idx}] DONE variant={variant} contract={len(contract)}ch "
          f"success={success} calls={n_calls} "
          f"tokens(p={prompt_tokens} c={completion_tokens}) "
          f"elapsed={time.time()-_t_start:.1f}s", flush=True)
    return pair, full


def load_questions(path, doc_filter=None):
    qs = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except Exception:
                continue
            if doc_filter and o.get("doc_type") != doc_filter:
                continue
            qs.append({
                "id": i,
                "doc_type": o.get("doc_type"),
                "variant": o.get("variant"),
                "source": o.get("source"),
                "question": o.get("question", ""),
            })
    return qs


def load_done(out_dir):
    p = out_dir / "qa_pairs.jsonl"
    done = set()
    if p.exists():
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except Exception:
                continue
            # only treat non-empty, error-free rows as completed; empties/errors
            # are retried on the next launch (resume).
            if "id" in o and not o.get("empty") and "error" not in o:
                done.add(o["id"])
    return done


def _collect_test_models(env):
    """TEST_MODEL1..N env entries (via .env or os.environ), in numeric order."""
    found = []
    for k, v in env.items():
        m = re.fullmatch(r"TEST_MODEL(\d+)", k)
        if m and isinstance(v, str) and v.strip():
            found.append((int(m.group(1)), v.strip()))
    return [v for _, v in sorted(found)]


def _slug(name):
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-") or "model"


def main() -> int:
    env = R.load_env(HERE / ".." / ".." / ".env")
    client = OpenAI(base_url=env["TEACHER_BASE_URL"], api_key=env["TEACHER_API_KEY"])
    base_model = env["TEACHER_MODEL"]
    temperature = float(env.get("TEACHER_TEMPERATURE", "0.2"))
    extra_body = None
    if env.get("TEACHER_EXTRA_BODY"):
        try:
            extra_body = json.loads(env["TEACHER_EXTRA_BODY"])
        except Exception:
            extra_body = None
    endpoint = os.environ.get(
        "RETRIEVAL_ENDPOINT",
        "http://127.0.0.1:10100")
    qa_workers = int(os.environ.get("QA_WORKERS", "5"))
    tool_cap = int(os.environ.get("QA_TOOL_CAP", "6"))
    qa_runs = int(os.environ.get("QA_RUNS", "0") or "0")
    qfile = os.environ.get("QA_QUESTIONS", "outputs/merged_question_set/questions_merged.jsonl")
    doc_filter = os.environ.get("QA_DOC_TYPE") or None
    qa_out = os.environ.get("QA_OUT_DIR", "outputs/qa_pairs")

    # Sweep plan: TEST_MODEL1..N -> one pass each, output dir suffixed with the
    # model slug. No TEST_MODEL* -> single legacy pass with TEACHER_MODEL.
    test_models = _collect_test_models(env)
    sweeps = [(m, Path(f"{qa_out}_{_slug(m)}")) for m in test_models] \
        or [(base_model, Path(qa_out))]
    for m, od in sweeps:
        print(f"[qa] sweep model={m} out_dir={od}", flush=True)
    print(f"[qa] endpoint={endpoint[:60]}... workers={qa_workers} "
          f"tool_cap={tool_cap}", flush=True)

    questions = load_questions(qfile, doc_filter)
    if not questions:
        print("[qa] no questions loaded", flush=True)
        return 1

    # warm up retrieval once
    try:
        R.RemoteRetrieval(endpoint).get_template(questions[0]["doc_type"])
        print("[qa] endpoint warm.", flush=True)
    except Exception as e:
        print(f"[qa] warmup warning: {e}", flush=True)

    for model, out_dir in sweeps:
        out_dir.mkdir(parents=True, exist_ok=True)
        done = load_done(out_dir)
        pending = [q for q in questions if q["id"] not in done]
        print(f"[qa] model={model}: loaded {len(questions)} questions, "
              f"{len(done)} done, {len(pending)} pending", flush=True)
        if qa_runs:
            pending = pending[:qa_runs]
        if not pending:
            print(f"[qa] model={model}: nothing to do", flush=True)
            continue

        pairs = []
        with ThreadPoolExecutor(max_workers=qa_workers) as ex:
            futs = {}
            for i, q in enumerate(pending):
                if i > 0 and i < qa_workers:
                    time.sleep(3)
                futs[ex.submit(run_qa, q, client, model, endpoint, tool_cap,
                              temperature, extra_body)] = q
            for fut in as_completed(futs):
                q = futs[fut]
                try:
                    pair, full = fut.result()
                except Exception as e:
                    print(f"[qa{q['id']}] ERROR: {e}", flush=True)
                    pair = {"id": q["id"], "doc_type": q["doc_type"], "variant": q["variant"],
                            "source": q["source"], "question": q["question"], "reasoning": "",
                            "answer": "", "n_tool_calls": 0, "n_successful": 0,
                            "empty": True, "error": str(e)}
                    full = pair
                with _write_lock:
                    with open(out_dir / "qa_pairs.jsonl", "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(pair, ensure_ascii=False) + "\n")
                    with open(out_dir / "qa_pairs_full.jsonl", "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(full, ensure_ascii=False) + "\n")
                pairs.append(pair)

        empties = sum(1 for p in pairs if p.get("empty"))
        manifest = {"model": model, "endpoint": endpoint, "tool_cap": tool_cap,
                    "workers": qa_workers, "processed": len(pairs), "empty": empties,
                    "out_dir": str(out_dir)}
        (out_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[qa] DONE model={model} processed={len(pairs)} empty={empties} "
              f"-> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
