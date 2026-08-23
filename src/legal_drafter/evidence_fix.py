"""Trial runner: regenerate missing ``evidence_report`` (schema §2) via teacher.

Picks tasks that have NO usable ``evidence_report`` and are not recoverable
from any existing ``*_evidence*.jsonl`` file (the "nowhere" set), then asks
the teacher model to produce a complete evidence_report JSON for each.

This is the 10-task smoke test of the regeneration approach. Output is a
single machine-readable JSON file; a plain-text log is written alongside
(NO progress bars).

Run detached, e.g.:
    nohup PYTHONPATH=src .venv/bin/python -m legal_drafter.evidence_fix \
        --n 10 --out outputs/evidence_fix/trial10.json \
        --log outputs/evidence_fix/trial10.log > outputs/evidence_fix/launch.log 2>&1 &
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import datetime
from collections import defaultdict

# Import our configured teacher client (reads TEACHER_* from .env).
from legal_drafter.config import load_teacher

BASE = "dataset/03_drafting_tasks_sources/tasks/task_pool_v1.jsonl"
EVIDENCE_GLOBS = [
    "dataset/03_drafting_tasks_sources/tasks/*evidence*.jsonl",
    "dataset/03_drafting_tasks_sources/versioned_task_source_sets/**/*evidence*.jsonl",
]

SYSTEM_PROMPT = (
    "You are a Polish legal-regime classifier for a legal-drafting "
    "distillation dataset. You are given a drafting task (an instruction, "
    "structured facts, and the retrieved statute sources). Your job is to "
    "produce a COMPLETE `evidence_report` JSON that maps the task to the "
    "legal norms required to draft it. Output ONLY a single valid JSON "
    "object. No commentary, no markdown fences."
)


def _schema_skeleton() -> str:
    return """{
  "task_id": "<same as input task_id>",
  "topic": "<task topic>",
  "regime": {
    "primary_regime": "<e.g. zlecenie_general, ordinary_residential_lease>",
    "doc_type": "<MUST equal facts.doc_type>",
    "topic": "<task topic>",
    "confidence": "high|medium|low",
    "required_source_roles": ["applicable_rule"],
    "forbidden_regimes": [],
    "must_use_concepts": ["<Polish legal concept that MUST appear in the draft>"],
    "forbidden_concepts": ["<Polish legal concept that MUST NOT appear>"],
    "notes": ["<string note>", "..."]
  },
  "legal_issues": [
    {
      "issue_id": "<topic_focus>",
      "clause_functions": ["<function the clause must perform>"],
      "must_support": ["<norm_id that is REQUIRED, from provided sources>"],
      "may_support": ["<norm_id that may support, from provided sources>"],
      "allowed_acts": ["<act_id, e.g. KC>"]
    }
  ],
  "required_norms": ["<norm_id required to support the draft>"],
  "covered_required_norms": ["<required_norms that are present in provided sources>"],
  "missing_required_norms": ["<required_norms NOT present in provided sources>"],
  "selected_norms": ["<norm_ids you selected from the provided sources>"],
  "selected_count": <int>,
  "eligible_count": <int>,
  "negative_count": <int>
}"""


def build_user_prompt(task: dict) -> str:
    facts = task.get("facts", {})
    sources = task.get("sources", [])
    src_block = []
    for s in sources:
        norms = s.get("norm_ids") or []
        src_block.append(
            f"- source_ref={s.get('source_ref')} | norm_ids={norms} | "
            f"article={s.get('article_number')} | act_id={s.get('act_id')} | "
            f"evidence_role={s.get('evidence_role')}\n  text: {s.get('text','')}"
        )
    src_text = "\n".join(src_block)
    return (
        f"TASK_ID: {task.get('task_id')}\n"
        f"TOPIC: {task.get('topic')}\n"
        f"INSTRUCTION: {task.get('instruction')}\n"
        f"FACTS: {json.dumps(facts, ensure_ascii=False)}\n\n"
        f"RETRIEVED SOURCES (use ONLY these; norm_ids you may cite):\n{src_text}\n\n"
        "Produce the evidence_report JSON with these HARD constraints:\n"
        "1. regime.doc_type MUST equal the facts.doc_type above.\n"
        "2. selected_norms and required_norms MUST be drawn ONLY from the "
        "norm_ids listed in RETRIEVED SOURCES. Do not invent norm_ids.\n"
        "3. legal_issues[].must_support / may_support MUST be norm_ids from "
        "the provided sources.\n"
        "4. notes is a JSON ARRAY of strings (never a string).\n"
        "5. confidence is one of high|medium|low.\n"
        "6. covered_required_norms = required_norms present in sources; "
        "missing_required_norms = required_norms NOT in sources.\n"
        "7. selected_count = len(selected_norms); eligible_count = number of "
        "distinct norm_ids available in sources; negative_count = total "
        "available source chunks (or 0 if unknown).\n\n"
        f"JSON schema to produce:\n{_schema_skeleton()}\n\n"
        "Output ONLY the JSON object."
    )


def extract_json(text: str):
    if text is None:
        return None
    text = text.strip()
    # strip markdown fences if present
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
    try:
        return json.loads(text)
    except Exception:
        pass
    # fallback: first { ... last }
    a, b = text.find("{"), text.rfind("}")
    if a != -1 and b != -1 and b > a:
        try:
            return json.loads(text[a : b + 1])
        except Exception:
            return None
    return None


def validate(ev: dict, task: dict) -> dict:
    issues = []
    if not isinstance(ev, dict):
        return {"ok": False, "issues": ["not a JSON object"]}
    regime = ev.get("regime")
    if not isinstance(regime, dict):
        issues.append("regime missing/not object")
    else:
        if regime.get("doc_type") != task.get("facts", {}).get("doc_type"):
            issues.append("regime.doc_type != facts.doc_type")
        if regime.get("confidence") not in ("high", "medium", "low"):
            issues.append("confidence not in high|medium|low")
        if not isinstance(regime.get("notes", []), list):
            issues.append("regime.notes not a list")
        if not isinstance(regime.get("must_use_concepts", []), list):
            issues.append("must_use_concepts not a list")
    # norms must resolve to task sources
    avail = set()
    for s in task.get("sources", []):
        avail.update(s.get("norm_ids") or [])
    for field in ("selected_norms", "required_norms", "covered_required_norms", "missing_required_norms"):
        vals = ev.get(field)
        if vals is None:
            issues.append(f"{field} missing")
        elif not isinstance(vals, list):
            issues.append(f"{field} not a list")
        else:
            bad = [v for v in vals if v not in avail]
            if bad:
                issues.append(f"{field} has unresolvable norm_ids: {bad}")
    li = ev.get("legal_issues")
    if not isinstance(li, list) or not li:
        issues.append("legal_issues missing/empty")
    else:
        for i, x in enumerate(li):
            for f in ("must_support", "may_support"):
                for v in x.get(f, []) or []:
                    if v not in avail:
                        issues.append(f"legal_issues[{i}].{f} unresolvable: {v}")
    return {"ok": len(issues) == 0, "issues": issues}


def load_tasks(path: str) -> dict:
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                o = json.loads(line)
                out[o.get("task_id")] = o
    return out


def has_ev(o: dict) -> bool:
    er = o.get("evidence_report")
    return bool(er and isinstance(er, dict) and er.get("regime"))


def covered_set() -> set:
    ids = set()
    files = []
    for pat in EVIDENCE_GLOBS:
        files.extend(glob.glob(pat, recursive=True))
    for p in set(files):
        try:
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    o = json.loads(line)
                    er = o.get("evidence_report")
                    if er and isinstance(er, dict) and er.get("regime"):
                        ids.add(o.get("task_id"))
        except Exception as e:
            print(f"[warn] could not read {p}: {e}", file=sys.stderr)
    return ids


def pick_tasks(base: dict, nowhere: list, n: int) -> list:
    groups = defaultdict(list)
    for tid in nowhere:
        key = (base[tid].get("facts", {}).get("doc_type"), base[tid].get("topic"))
        groups[key].append(tid)
    sel = []
    while len(sel) < n and groups:
        for k in list(groups.keys()):
            if groups[k] and len(sel) < n:
                sel.append(groups[k].pop(0))
            if not groups[k]:
                del groups[k]
    return sel


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--out", default="outputs/evidence_fix/trial10.json")
    ap.add_argument("--log", default="outputs/evidence_fix/trial10.log")
    ap.add_argument("--base", default=BASE)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    log = open(args.log, "w")

    def logline(s):
        print(s, flush=True)
        print(s, file=log, flush=True)

    logline(f"[{datetime.datetime.utcnow().isoformat()}] loading tasks from {args.base}")
    base = load_tasks(args.base)
    covered = covered_set()
    logline(f"base tasks={len(base)}; covered-by-other-evidence={len(covered)}")

    missing = [t for t, o in base.items() if not has_ev(o)]
    nowhere = [t for t in missing if t not in covered]
    logline(f"missing evidence_report in base={len(missing)}; nowhere(non-recoverable)={len(nowhere)}")

    sel = pick_tasks(base, nowhere, args.n)
    logline(f"selected {len(sel)} tasks for generation")

    cfg = load_teacher()
    client = cfg.make_client()
    logline(f"teacher model={cfg.model} base_url={cfg.base_url}")

    results = []
    for i, tid in enumerate(sel, 1):
        task = base[tid]
        logline(f"[{i}/{len(sel)}] {tid} -> generating")
        user = build_user_prompt(task)
        try:
            resp = client.chat.completions.create(
                model=cfg.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                timeout=cfg.timeout,
                stream=True,
            )
            content = ""
            finish_reason = None
            for chunk in resp:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    content += delta.content
                if chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason
            if not content:
                logline(f"[{i}] {tid} -> EMPTY CONTENT (finish={finish_reason})")
                results.append({"task_id": tid, "ok": False, "error": "empty content"})
                continue
            ev = extract_json(content)
            if ev is None:
                results.append({"task_id": tid, "ok": False, "error": "no JSON parsed", "raw": content})
                logline(f"[{i}] {tid} -> PARSE FAIL; starts_with_fence={content.lstrip().startswith('```')}; raw={content[:1200]!r}")
                continue
            v = validate(ev, task)
            results.append({"task_id": tid, "ok": v["ok"], "evidence_report": ev, "validation": v})
            logline(f"[{i}] {tid} -> {'OK' if v['ok'] else 'INVALID'}: {v['issues']}")
        except Exception as e:
            results.append({"task_id": tid, "ok": False, "error": str(e)})
            logline(f"[{i}] {tid} -> ERROR: {e!r}")

    out = {
        "meta": {
            "n": len(sel),
            "model": cfg.model,
            "generated_at": datetime.datetime.utcnow().isoformat(),
            "source": args.base,
            "nowhere_total": len(nowhere),
        },
        "results": results,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    logline(f"DONE. wrote {args.out} ({len(results)} results)")
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
