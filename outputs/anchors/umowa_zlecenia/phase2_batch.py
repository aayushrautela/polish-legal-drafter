"""Phase 2 (BATCHED + SINGLE BATCH TOOL) - LLM add/remove mutation, grounded in RAG.

Anchors are grouped by doc_type so the template is paid ONCE, then ~BATCH anchors
are packed into a single LLM call. Instead of relying on flaky PARALLEL tool
calls (proxies trim/drop multi-index deltas), the model is FORCED to make exactly
ONE tool call - submit_mutations - whose single `mutations` array carries every
anchor's ops. One args blob, one index, no stitching. This mirrors the standard
"forced tool = structured output" pattern used before/alongside strict
structured outputs.

Usage:
  PYTHONPATH=src RETRIEVAL_ENDPOINT=... .venv/bin/python -m legal_drafter.phase2_batch \
      --in  outputs/anchors/anchor_pool.phase1.jsonl \
      --out outputs/anchors/anchor_pool.phase2.jsonl \
      --batch 5 --related 2 --parallel 4 --retries 3
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from legal_drafter.config import load_checker  # noqa: E402

TRUNC_TEMPLATE = 2600
TRUNC_ANCHOR = 1500
TRUNC_RELATED = 850

TOOL_NAME = "submit_mutations"

SYSTEM_TOOLS = (
    "You are a meticulous Polish legal expert building a question bank of "
    "layperson instructions. You receive a shared contract TEMPLATE (when present) "
    "and several ANCHOR situations labelled A1..An. Each anchor has a real "
    "RAG-retrieved anchor clause plus 1-2 REFERENCE clauses (A{i}R1..) also from "
    "RAG, each with a source_ref.\n"
    "Call the submit_mutations tool EXACTLY ONCE with a mutations array covering "
    "EVERY anchor A1..An. Each element: {anchor_id, adds, removes} where\n"
    "- adds: 1-3 items {condition, cite_ref}. condition = one NEW concrete "
    "obligation/condition grounded in that anchor's supplied clauses. cite_ref "
    "MUST be one of that anchor's labels (A{i}, A{i}R1, A{i}R2) or its "
    "source_ref. Never invent a reference.\n"
    "- removes: 0-2 items {template_clause}; quote an exact template fragment "
    "ONLY when the anchor clearly overrides it.\n"
    "STRICT QUALITY RULES (violations make the data useless):\n"
    "1. Do NOT restate or paraphrase the anchor clause as an add. Each add must "
    "be a NEW, DISTINCT condition the layperson would care about (a concrete "
    "obligation, deadline, notice period, penalty, deposit, warranty, etc.), not "
    "filler added just to reach the count.\n"
    "2. Do NOT add anything already stated in the anchor or in the template.\n"
    "3. Only cite a REFERENCE clause (A{i}R1..) when DIRECTLY RELEVANT to the "
    "anchor situation; citing the anchor (A{i}) is fine when the new condition "
    "flows from it.\n"
    "4. If an anchor genuinely needs fewer than 2 operations, emit only what is "
    "substantive — quality over quantity.\n"
    "Respond ONLY with the single submit_mutations call, no prose."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "description": (
                "Submit ALL mutation operations for EVERY anchor (A1..An) in one "
                "call. Exactly one call per response."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "mutations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "anchor_id": {
                                    "type": "string",
                                    "description": "anchor id, e.g. A1",
                                },
                                "adds": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "condition": {
                                                "type": "string",
                                                "description": "opis warunku po polsku",
                                            },
                                            "cite_ref": {
                                                "type": "string",
                                                "description": (
                                                    "label tej kotwicy: A{i}, "
                                                    "A{i}R1.. lub source_ref"),
                                            },
                                        },
                                        "required": ["condition", "cite_ref"],
                                    },
                                },
                                "removes": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "template_clause": {
                                                "type": "string",
                                                "description": "fragment klauzuli z szablonu",
                                            },
                                        },
                                        "required": ["template_clause"],
                                    },
                                },
                            },
                            "required": ["anchor_id", "adds", "removes"],
                        },
                    },
                },
                "required": ["mutations"],
            },
        },
    },
]


def _trunc(text: str, n: int) -> str:
    if text is None:
        return ""
    text = text.strip()
    return text if len(text) <= n else text[:n - 1].rstrip() + "…"


def _chat_batch_tool(client, cfg, messages, max_tokens: int,
                     temperature: float = 0.2):
    """Force ONE submit_mutations call; returns its parsed arguments dict.

    Streams the single args blob (index 0 only), so no multi-index delta
    stitching is involved. If the backend rejects the forced tool_choice we
    retry once with 'auto' plus a reminder line.
    """
    base = dict(model=cfg.model, messages=messages, tools=TOOLS,
                temperature=temperature, max_tokens=max_tokens, stream=True)
    if cfg.extra_body:
        base["extra_body"] = cfg.extra_body
    if cfg.extra_headers:
        base["extra_headers"] = cfg.extra_headers

    def accumulate(kwargs) -> tuple[str | None, str]:
        args_buf = ""
        name = None
        for chunk in client.chat.completions.create(**kwargs):
            if not chunk.choices:
                continue
            tcs = chunk.choices[0].delta.tool_calls
            if not tcs:
                continue
            for t in tcs:
                if t.function and t.function.name:
                    name = t.function.name
                if t.function and t.function.arguments:
                    args_buf += t.function.arguments
        return name, args_buf

    attempts = [
        dict(base, tool_choice={"type": "function", "function": {"name": TOOL_NAME}}),
        dict(base, tool_choice="auto",
             messages=messages + [{"role": "user",
                                   "content": f"Now call {TOOL_NAME} once with "
                                              "every anchor's mutations."}]),
    ]
    last_err: Exception | None = None
    for kwargs in attempts:
        try:
            name, args_buf = accumulate(kwargs)
            if not args_buf.strip():
                last_err = RuntimeError("empty tool arguments")
                continue
            try:
                parsed = json.loads(args_buf)
            except Exception:
                # salvage common truncation artifacts (trailing garbage)
                m = re.search(r"\{.*\}", args_buf, re.DOTALL)
                parsed = json.loads(m.group(0)) if m else {}
                if not isinstance(parsed, dict) or not parsed:
                    raise ValueError(f"unparseable arguments ({len(args_buf)} chars)")
            if name and name != TOOL_NAME and isinstance(parsed, list):
                parsed = {"mutations": parsed}
            if not isinstance(parsed, dict):
                raise ValueError("arguments are not an object")
            return parsed
        except TypeError as e:
            last_err = e
            continue
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
    raise last_err if last_err else RuntimeError("no tool call produced")


def build_batch_prompt(batch: list[dict], template_text: str | None) -> str:
    parts = []
    if template_text:
        parts.append("[TEMPLATE]\n" + _trunc(template_text, TRUNC_TEMPLATE))
    else:
        parts.append("[TEMPLATE]\n(no shared template for this group)")
    for i, rec in enumerate(batch, 1):
        related = (rec.get("related_clauses") or [])[:2]
        block = (f"[ANCHOR A{i}] chunk_id={rec['chunk_id']}\n"
                 f"Anchor clause (label A{i}, source_ref={rec.get('source_ref')}):\n"
                 f"{_trunc(rec.get('text') or '', TRUNC_ANCHOR)}")
        for j, c in enumerate(related, 1):
            block += (f"\nReference R{j} for A{i} (label A{i}R{j}, "
                      f"source_ref={c.get('source_ref')}):\n"
                      f"{_trunc(c.get('text') or '', TRUNC_RELATED)}")
        parts.append(block)
    return ("\n\n".join(parts)
            + f"\n\nCall submit_mutations ONCE with a mutations array covering "
              f"EVERY anchor A1..A{len(batch)}. Each element needs anchor_id plus "
              "its adds/removes; every add cites exactly one of that anchor's "
              "labels.")


def _labels_for(i: int, rec: dict, related: list[dict]) -> dict:
    # never map a label to None/"" - null refs produced ungrounded adds
    anchor_ref = rec.get("source_ref") or rec.get("chunk_id")
    lab = {f"A{i}": anchor_ref}
    for j, c in enumerate(related, 1):
        ref = c.get("source_ref") or c.get("chunk_id")
        if not ref:
            continue
        lab[f"A{i}R{j}"] = ref
    for k, v in list(lab.items()):
        if v:
            lab[v] = v
    return lab


def _valid_add(a, labels) -> bool:
    cond = (a.get("condition") or "").strip()
    ref = (a.get("cite_ref") or "").strip()
    return len(cond) >= 12 and bool(ref) and ref in labels and bool(labels[ref])


def _validate_ops(adds: list, removes: list, labels: dict):
    v_adds, v_removes = [], []
    for a in adds or []:
        cond = (a.get("condition") or "").strip()
        ref = (a.get("cite_ref") or "").strip()
        if not _valid_add(a, labels):
            continue
        v_adds.append({"condition": cond, "cite_ref": labels[ref]})
    for r in removes or []:
        frag = (r.get("template_clause") or "").strip()
        if len(frag) < 8:
            continue
        v_removes.append({"template_clause": frag})
    return v_adds[:4], v_removes[:4]


def process_batch(client, cfg, batch: list[dict], template_text: str | None,
                  related_k: int, max_tokens: int, retries: int) -> dict:
    anchor_labels = {}
    for i, rec in enumerate(batch, 1):
        related = (rec.get("related_clauses") or [])[:related_k]
        anchor_labels[i] = _labels_for(i, rec, related)
    prompt = build_batch_prompt(batch, template_text)
    messages = [{"role": "system", "content": SYSTEM_TOOLS},
                {"role": "user", "content": prompt}]
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            parsed = _chat_batch_tool(client, cfg, messages, max_tokens)
            by_anchor = {}
            muts = parsed.get("mutations")
            if isinstance(muts, dict):  # single-anchor shorthand
                muts = [muts]
            for m in muts or []:
                if not isinstance(m, dict):
                    continue
                aid = str(m.get("anchor_id") or "").strip()
                mo = re.match(r"A?(\d+)", aid)
                if not mo:
                    continue
                by_anchor.setdefault(int(mo.group(1)), []).append(m)
            res = {}
            for i, rec in enumerate(batch, 1):
                entries = by_anchor.get(i, [])
                if not entries:
                    res[rec["chunk_id"]] = {"adds": [], "removes": [],
                                            "mutation_error":
                                            "no mutations for anchor"}
                    continue
                adds, removes = [], []
                for m in entries:
                    a, r = _validate_ops(m.get("adds"), m.get("removes"),
                                         anchor_labels[i])
                    adds += a
                    removes += r
                res[rec["chunk_id"]] = ({"adds": adds, "removes": removes}
                                        if adds or removes else
                                        {"adds": [], "removes": [],
                                         "mutation_error": "no valid ops"})
            return res
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(min(2 * attempt, 10))
    return {rec["chunk_id"]: {"adds": [], "removes": [],
                              "mutation_error": last_err or "unknown"}
            for rec in batch}


def _rewrite_line(out: Path, chunk_id: str, new_rec: dict) -> None:
    lines = out.read_text(encoding="utf-8").splitlines()
    out_lines = []
    replaced = False
    for l in lines:
        if not l.strip():
            continue
        try:
            r = json.loads(l)
        except Exception:
            out_lines.append(l)
            continue
        if r.get("chunk_id") == chunk_id and not replaced:
            out_lines.append(json.dumps(new_rec, ensure_ascii=False))
            replaced = True
        else:
            out_lines.append(l)
    if not replaced:
        out_lines.append(json.dumps(new_rec, ensure_ascii=False))
    out.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="outputs/anchors/anchor_pool.phase1.jsonl")
    ap.add_argument("--out", default="outputs/anchors/anchor_pool.phase2.jsonl")
    ap.add_argument("--batch", type=int, default=5)
    ap.add_argument("--related", type=int, default=2)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--retry-pass", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=12000)
    args = ap.parse_args()

    cfg = load_checker()
    client = cfg.make_client()
    inp = Path(args.inp)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    recs = [json.loads(l) for l in inp.read_text(encoding="utf-8").splitlines() if l.strip()]
    by_id = {r["chunk_id"]: r for r in recs}
    done = set()
    if out.exists():
        for l in out.read_text(encoding="utf-8").splitlines():
            if l.strip():
                try:
                    r = json.loads(l)
                    if not r.get("mutation_error"):
                        done.add(r["chunk_id"])
                except Exception:
                    pass
    pending = [r for r in recs if r["chunk_id"] not in done]
    print(f"records={len(recs)} done={len(done)} pending={len(pending)}", flush=True)

    groups = defaultdict(list)
    for r in pending:
        groups[r.get("doc_type_primary") or "other"].append(r)

    batches = []
    for dt, items in groups.items():
        template_text = items[0].get("template_text") if dt != "other" else None
        for k in range(0, len(items), args.batch):
            batches.append((dt, template_text, items[k:k + args.batch]))
    print(f"doc_types={len(groups)} batches={len(batches)}", flush=True)

    write_lock = Lock()
    errs = 0
    written = 0
    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futs = {ex.submit(process_batch, client, cfg, b[2], b[1], args.related,
                          args.max_tokens, args.retries): b for b in batches}
        completed = 0
        for fut in as_completed(futs):
            dt, template_text, batch = futs[fut]
            try:
                res = fut.result()
            except Exception as e:  # noqa: BLE001
                res = {r["chunk_id"]: {"adds": [], "removes": [],
                                       "mutation_error": str(e)} for r in batch}
            with write_lock:
                with out.open("a", encoding="utf-8") as f:
                    for rec in batch:
                        cid = rec["chunk_id"]
                        mut = res.get(cid, {"adds": [], "removes": [],
                                            "mutation_error": "missing"})
                        out_rec = dict(rec)
                        out_rec["adds"] = mut.get("adds", [])
                        out_rec["removes"] = mut.get("removes", [])
                        if mut.get("mutation_error"):
                            out_rec["mutation_error"] = mut["mutation_error"]
                            errs += 1
                        out_rec["phase2_dt"] = dt
                        out_rec["mutation_source"] = "batch_tool"
                        f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                        written += 1
            completed += 1
            if completed % 5 == 0 or completed == len(batches):
                print(f"progress batches {completed}/{len(batches)} "
                      f"written={written} errors={errs}", flush=True)

    # Retry pass: any anchor left with no mutations gets a single-anchor call,
    # which the model covers reliably (one anchor -> focused output).
    errored = []
    if out.exists():
        for l in out.read_text(encoding="utf-8").splitlines():
            if not l.strip():
                continue
            try:
                r = json.loads(l)
            except Exception:
                continue
            if r.get("mutation_error"):
                errored.append(r["chunk_id"])
    print(f"retry pass: {len(errored)} errored anchors", flush=True)
    rdone = 0
    for cid in errored:
        rec = by_id.get(cid)
        if not rec:
            continue
        tmpl = rec.get("template_text") if (rec.get("doc_type_primary") and rec.get("doc_type_primary") != "other") else None
        mut = None
        for _ in range(max(1, args.retry_pass)):
            res = process_batch(client, cfg, [rec], tmpl, args.related,
                                args.max_tokens, 1)
            mut = res.get(cid)
            if mut and not mut.get("mutation_error"):
                break
        if mut is None:
            mut = {"adds": [], "removes": [], "mutation_error": "retry failed"}
        out_rec = dict(rec)
        out_rec["adds"] = mut.get("adds", [])
        out_rec["removes"] = mut.get("removes", [])
        if mut.get("mutation_error"):
            out_rec["mutation_error"] = mut["mutation_error"]
        else:
            out_rec.pop("mutation_error", None)
        out_rec["phase2_dt"] = rec.get("doc_type_primary") or "other"
        out_rec["mutation_source"] = "batch_tool_retry"
        _rewrite_line(out, cid, out_rec)
        rdone += 1
        if rdone % 10 == 0 or rdone == len(errored):
            print(f"retry progress {rdone}/{len(errored)}", flush=True)

    final_err = 0
    if out.exists():
        for l in out.read_text(encoding="utf-8").splitlines():
            if l.strip():
                try:
                    if json.loads(l).get("mutation_error"):
                        final_err += 1
                except Exception:
                    pass
    print(f"DONE. written={written} first_pass_errors={errs} "
          f"retried={len(errored)} final_errors={final_err}", flush=True)


if __name__ == "__main__":
    main()
