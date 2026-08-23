"""Phase 2: LLM mutation of each anchor's conditions (grounded in RAG).

One streamed LLM step per anchor - the DECIDE step: the CHECKER_MODEL reasons
about the template + anchor + supplied RAG reference clauses and returns a
concrete plan of 2-4 operations (additions grounded in a supplied clause, and/or
removals of template fragments). This single streamed response already contains
the several operations (the model "decides what to add/remove" up front), so it
is the multiple-mutation output - no separate single-call tool step is needed.
Every ADD is validated against the supplied RAG clauses (label R1..Rn/ANCHOR or
source_ref); every REMOVE must quote a real template fragment. Resume + parallel
+ retry.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from legal_drafter.config import load_checker


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "add_condition",
            "description": (
                "Add ONE new condition/clause to the document, grounded in exactly one "
                "of the supplied RAG clauses. Cite its label (R1..Rn or ANCHOR) or its "
                "source_ref. Call multiple times to add several conditions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "condition": {"type": "string", "description": "opis warunku po polsku"},
                    "cite_ref": {
                        "type": "string",
                        "description": "etykieta klauzuli: R1..Rn lub ANCHOR, albo source_ref",
                    },
                },
                "required": ["condition", "cite_ref"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remove_condition",
            "description": "Remove ONE existing template clause by quoting its real fragment. Call multiple times.",
            "parameters": {
                "type": "object",
                "properties": {
                    "template_clause": {"type": "string", "description": "fragment klauzuli z szablonu"}
                },
                "required": ["template_clause"],
            },
        },
    },
]

SYSTEM_DECISION = (
    "You are a Polish legal expert building a dataset of realistic layperson legal "
    "requests. You are given a base TEMPLATE (its clauses), an ANCHOR clause (a real "
    "corpus clause), and REFERENCE clauses R1..Rn (real corpus chunks). DECIDE a concrete "
    "mutation of the template's conditions. Produce 2 to 4 operations in total: add new "
    "conditions grounded in the supplied clauses (cite the clause label R1..Rn or ANCHOR, "
    "or its source_ref), and/or remove existing template clauses (quote the real template "
    "fragment). Think about which supplied clauses are relevant to the anchor situation, "
    "then choose several operations. Return ONLY strict JSON: "
    '{"adds":[{"condition":str,"cite_ref":str}],"removes":[{"template_clause":str}]}. '
    "Every add's cite_ref MUST be one of R1..Rn / ANCHOR or a supplied source_ref."
)

SYSTEM_ACT = (
    "You are implementing a pre-decided mutation plan as tool calls. Emit ONE "
    "add_condition tool call per planned addition and ONE remove_condition tool call per "
    "planned removal, matching the plan exactly. Respond ONLY with tool calls, no prose."
)


def _trunc(s, n):
    s = s or ""
    return s if len(s) <= n else s[:n] + " \u2026"


def _build_context(rec):
    anchor = rec.get("text") or ""
    tmpl = rec.get("template_text")
    rel = rec.get("related_clauses") or []
    labels = {}
    parts = []
    if anchor.strip():
        labels["ANCHOR"] = {
            "source_ref": rec.get("source_ref"),
            "chunk_id": rec.get("chunk_id"),
            "text": _trunc(anchor, 1800),
        }
        parts.append(f"[ANCHOR] source_ref={rec.get('source_ref')}\n{_trunc(anchor, 1800)}")
    for i, c in enumerate(rel, 1):
        lid = f"R{i}"
        labels[lid] = {
            "source_ref": c.get("source_ref"),
            "chunk_id": c.get("chunk_id"),
            "text": _trunc(c.get("text") or "", 1000),
        }
        parts.append(
            f"[{lid}] source_ref={c.get('source_ref')} title={c.get('title')}\n"
            f"{_trunc(c.get('text') or '', 1000)}"
        )
    if tmpl and tmpl.strip():
        tblock = f"[TEMPLATE]\n{_trunc(tmpl, 2500)}"
    else:
        tblock = "[TEMPLATE]\nNO TEMPLATE (add-only mode)"
    user = f"{tblock}\n\n{chr(10).join(parts)}\n\nDecide the mutation and return JSON."
    return (
        [
            {"role": "system", "content": SYSTEM_DECISION},
            {"role": "user", "content": user},
        ],
        labels,
    )


def _lookup(labels):
    lookup = {}
    for lid, info in labels.items():
        lookup[lid] = info
        if info.get("source_ref"):
            lookup[str(info["source_ref"])] = info
        if info.get("chunk_id"):
            lookup[str(info["chunk_id"])] = info
    return lookup


def _validate_plan(plan, labels):
    if not isinstance(plan, dict):
        return None, "plan not an object"
    adds = plan.get("adds") or []
    removes = plan.get("removes") or []
    if not isinstance(adds, list) or not isinstance(removes, list):
        return None, "adds/removes not lists"
    lookup = _lookup(labels)
    out_adds, out_removes = [], []
    for a in adds:
        if not isinstance(a, dict):
            return None, "bad add"
        ref = a.get("cite_ref")
        cond = a.get("condition")
        info = lookup.get(ref) if ref is not None else None
        if info is None:
            return None, f"cite_ref {ref!r} not supplied"
        if not str(cond or "").strip():
            return None, "empty condition"
        out_adds.append(
            {
                "condition": str(cond).strip(),
                "cite_ref": ref,
                "source_ref": info.get("source_ref"),
                "chunk_id": info.get("chunk_id"),
            }
        )
    for rm in removes:
        if not isinstance(rm, dict):
            return None, "bad remove"
        tc = rm.get("template_clause")
        if not str(tc or "").strip():
            return None, "empty template_clause"
        out_removes.append({"template_clause": str(tc).strip()})
    if not out_adds and not out_removes:
        return None, "no operations"
    return {"adds": out_adds, "removes": out_removes}, None


def _validate_calls(calls, labels):
    lookup = _lookup(labels)
    adds, removes = [], []
    for name, args in calls:
        if name == "add_condition":
            ref = args.get("cite_ref")
            cond = args.get("condition")
            info = lookup.get(ref) if ref is not None else None
            if info is None:
                return None, f"cite_ref {ref!r} not supplied"
            if not str(cond or "").strip():
                return None, "empty condition"
            adds.append(
                {
                    "condition": str(cond).strip(),
                    "cite_ref": ref,
                    "source_ref": info.get("source_ref"),
                    "chunk_id": info.get("chunk_id"),
                }
            )
        elif name == "remove_condition":
            tc = args.get("template_clause")
            if not str(tc or "").strip():
                return None, "empty template_clause"
            removes.append({"template_clause": str(tc).strip()})
        else:
            return None, f"unknown tool {name}"
    if not adds and not removes:
        return None, "no tool calls"
    return {"adds": adds, "removes": removes}, None


def _chat_content(client, cfg, messages, max_tokens=1500, temperature=0.3):
    kwargs = dict(
        model=cfg.model, messages=messages, temperature=temperature,
        max_tokens=max_tokens, stream=True,
    )
    if cfg.extra_body:
        kwargs["extra_body"] = cfg.extra_body
    if cfg.extra_headers:
        kwargs["extra_headers"] = cfg.extra_headers
    parts = []
    for chunk in client.chat.completions.create(**kwargs):
        if chunk.choices and chunk.choices[0].delta.content:
            parts.append(chunk.choices[0].delta.content)
    return "".join(parts)


def _chat_tools(client, cfg, messages, max_tokens=2000, temperature=0.2):
    kwargs = dict(
        model=cfg.model, messages=messages, tools=TOOLS, tool_choice="auto",
        temperature=temperature, max_tokens=max_tokens, stream=True,
    )
    if cfg.extra_body:
        kwargs["extra_body"] = cfg.extra_body
    if cfg.extra_headers:
        kwargs["extra_headers"] = cfg.extra_headers
    acc = {}
    for chunk in client.chat.completions.create(**kwargs):
        if not chunk.choices:
            continue
        tcs = chunk.choices[0].delta.tool_calls
        if not tcs:
            continue
        for t in tcs:
            d = acc.setdefault(t.index, {"id": None, "name": None, "args": ""})
            if t.id:
                d["id"] = t.id
            if t.function and t.function.name:
                d["name"] = t.function.name
            if t.function and t.function.arguments:
                d["args"] += t.function.arguments
    calls = []
    for idx in sorted(acc.keys()):
        d = acc[idx]
        if not d["name"]:
            continue
        try:
            args = json.loads(d["args"] or "{}")
        except Exception:
            args = {}
        calls.append((d["name"], args))
    return calls


def _extract_json(text):
    s = text.find("{"); e = text.rfind("}")
    if s < 0 or e < 0 or e <= s:
        return None
    try:
        return json.loads(text[s : e + 1])
    except Exception:
        return None


def _one(rec, client, cfg):
    has_tmpl = bool((rec.get("template_text") or "").strip())
    has_rel = bool(rec.get("related_clauses"))
    if not has_tmpl and not has_rel:
        return {**rec, "mutation": None, "mutation_error": "no template and no related clauses"}

    msgs, labels = _build_context(rec)
    # Step 1: DECIDE (streamed). The single decision response already contains
    # several operations (2-4), so it IS the multiple-mutation output. A separate
    # tool-call step proved unreliable (the model collapsed it to one call) and
    # doubled cost, so we use the decided, validated, RAG-grounded plan directly.
    content = _chat_content(client, cfg, msgs, max_tokens=1500, temperature=0.3)
    plan = _extract_json(content)
    if plan is None:
        raise RuntimeError("decision: no JSON")
    decided, err = _validate_plan(plan, labels)
    if err:
        raise RuntimeError(f"decision: {err}")
    rec = dict(rec)
    rec["mutation"] = decided
    rec["mutation_source"] = "decided_plan"
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="outputs/anchors/anchor_pool.phase1.jsonl")
    ap.add_argument("--out", default="outputs/anchors/anchor_pool.phase2.jsonl")
    ap.add_argument("--parallel", type=int, default=5)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    cfg = load_checker()
    client = cfg.make_client()
    random.seed(args.seed)

    recs = [json.loads(l) for l in open(args.inp)]
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    if r.get("mutation") or r.get("mutation_error"):
                        done.add(r.get("chunk_id"))
                except Exception:
                    continue
    pending = [r for r in recs if r.get("chunk_id") not in done]

    write_lock = threading.Lock()
    out_f = open(args.out, "a")
    stats = {"ok": 0, "error": 0, "from_plan": 0}
    done_n = len(done)

    def gen_one(rec):
        for attempt in range(1, max(args.retries, 1) + 1):
            try:
                return _one(rec, client, cfg)
            except Exception as exc:
                if attempt < args.retries:
                    time.sleep(min(2 ** attempt, 20))
        return {**rec, "mutation": None, "mutation_error": f"failed after {args.retries} attempts"}

    print(f"phase2: {len(pending)} pending of {len(recs)} (done={done_n}), parallel={args.parallel}")
    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futs = [ex.submit(gen_one, r) for r in pending]
        for fut in as_completed(futs):
            rec = fut.result()
            with write_lock:
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_f.flush()
                if rec.get("mutation"):
                    stats["ok"] += 1
                    if rec.get("mutation_source") == "decided_plan":
                        stats["from_plan"] += 1
                else:
                    stats["error"] += 1
                done_n += 1
                if done_n % 25 == 0 or done_n == len(recs):
                    print(f"progress {done_n}/{len(recs)} | ok={stats['ok']} err={stats['error']} plan_fallback={stats['from_plan']}")

    out_f.close()
    print(f"DONE. {args.out} | ok={stats['ok']} err={stats['error']} plan_fallback={stats['from_plan']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
