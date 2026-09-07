#!/usr/bin/env python3
"""Build the 58 external eval rows in SFT jsonl format and the merged 100-question set.

Reads:
  outputs/eval100/external_58_raw.json    (the 58 freshly authored questions)
  outputs/qa_pairs_all_final/sft_final_merged.jsonl    (for ref rows per doc_type)
  outputs/eval100/holdout_ids_true42.json    (the 42 ckpt250 holdout)

Writes:
  outputs/eval100/external_58.jsonl          (58 rows in SFT jsonl shape, ext_* ids)
  outputs/eval100/eval100_questions.jsonl   (42 holdout + 58 external = 100 rows)
  outputs/eval100/question_ids.json          (100 ids + doc_type + source)
  outputs/eval100/sampling_notes.md          (provenance, per-type counts, contamination log)

The 58 external rows reuse the system prompt of the SFT training set and a
reference user-role row from the same doc_type for scaffolding, with the
question text swapped. This is the minimum-friction way to feed the existing
eval_holdouts_v2_v3.py pipeline.

Contamination check (mandatory, see AGENTS.md):
  - id collision check against the entire SFT training pool (sft_final_merged.jsonl)
  - exact-question-text duplicate check
  - SequenceMatcher similarity against the closest training question (warn if >0.5)
"""
from __future__ import annotations

import json
import re
import difflib
from collections import Counter
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent
SFT = ROOT / "outputs" / "qa_pairs_all_final" / "sft_final_merged.jsonl"
HOLDOUT = ROOT / "outputs" / "eval100" / "holdout_ids_true42.json"
EXTERNAL_RAW = ROOT / "outputs" / "eval100" / "external_58_raw.json"

OUT_EXTERNAL = ROOT / "outputs" / "eval100" / "external_58.jsonl"
OUT_EVAL100 = ROOT / "outputs" / "eval100" / "eval100_questions.jsonl"
OUT_IDS = ROOT / "outputs" / "eval100" / "question_ids.json"
OUT_NOTES = ROOT / "outputs" / "eval100" / "sampling_notes.md"

EXTERNAL_SOURCE_TAG = "external_eval_v1"
SEED = 13

SYSTEM_PROMPT = (
    "You are a Polish legal drafter. Use the retrieval tools to find "
    "relevant legal provisions, then write a COMPLETE contract in Polish.\n\n"
    "Rules:\n"
    "- Use only provisions found in the retrieved chunks. Do not invent.\n"
    '- After researching, return ONLY strict JSON: {"summary": "<one-line '
    'scenario>", "contract": "<full contract>"}'
)


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f]


def contamination_check(external: list[dict], sft_rows: list[dict]) -> dict:
    """Id collision + exact text + max similarity vs SFT pool."""
    train_ids = {r["id"] for r in sft_rows}
    train_qs = [r["question"] for r in sft_rows]
    report: dict = {
        "id_collisions": [],
        "exact_duplicates": [],
        "max_similarity_per_question": [],
        "warnings": [],
    }
    for q in external:
        if q["id"] in train_ids:
            report["id_collisions"].append(q["id"])
        if q["question"] in train_qs:
            report["exact_duplicates"].append(q["id"])
        # find max similarity
        best = 0.0
        bestm = ""
        for tq in train_qs:
            r = difflib.SequenceMatcher(None, q["question"], tq).ratio()
            if r > best:
                best = r
                bestm = tq[:80]
        report["max_similarity_per_question"].append(
            {"id": q["id"], "doc_type": q["doc_type"], "max_sim": round(best, 3), "closest_train": bestm}
        )
        if best > 0.5:
            report["warnings"].append(q["id"])
    return report


def build_external_rows(external: list[dict], sft_rows: list[dict]) -> list[dict]:
    """Build SFT-format rows: each new question gets the canonical system prompt
    and a user-role message, plus a tools list copied from a reference row of
    the same doc_type (so any downstream consumer that reads tools from the
    row gets a sensible schema). The eval script's tool-schema patch in
    eval_holdouts_v2_v3.py rewrites the doc_type enum regardless.
    """
    ref_by_type: dict[str, dict] = {}
    for r in sft_rows:
        ref_by_type.setdefault(r["doc_type"], r)
    rows: list[dict] = []
    for q in external:
        ref = ref_by_type[q["doc_type"]]
        row = {
            "id": q["id"],
            "doc_type": q["doc_type"],
            "variant": "external",
            "source": EXTERNAL_SOURCE_TAG,
            "question": q["question"],
            "tools": list(ref.get("tools", [])),
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q["question"]},
            ],
        }
        rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    external_raw = json.load(open(EXTERNAL_RAW))
    external_qs = external_raw["questions"]
    sft_rows = load_jsonl(SFT)
    holdout = json.load(open(HOLDOUT))
    holdout_ids = holdout["ids"]

    # 1. contamination check
    contam = contamination_check(external_qs, sft_rows)
    if contam["id_collisions"] or contam["exact_duplicates"]:
        raise SystemExit(
            f"CONTAMINATION: id collisions={contam['id_collisions']} "
            f"exact_dupes={contam['exact_duplicates']}"
        )
    if contam["warnings"]:
        print("WARN: high similarity (>0.5) for:", contam["warnings"])

    # 2. build external rows
    external_rows = build_external_rows(external_qs, sft_rows)
    write_jsonl(OUT_EXTERNAL, external_rows)
    print(f"wrote {len(external_rows)} external rows -> {OUT_EXTERNAL}")

    # 3. merged 100-row set: 42 holdout + 58 external
    rows_by_id = {r["id"]: r for r in sft_rows}
    holdout_rows = [rows_by_id[i] for i in holdout_ids]
    eval100 = holdout_rows + external_rows
    # sanity: 100 rows, exactly 4 per doc_type
    assert len(eval100) == 100, len(eval100)
    by_type = Counter(r["doc_type"] for r in eval100)
    for dt, n in by_type.items():
        assert n == 4, (dt, n)
    write_jsonl(OUT_EVAL100, eval100)
    print(f"wrote {len(eval100)} merged rows -> {OUT_EVAL100}")

    # 4. question_ids.json: 100 ids with source
    qids = []
    for r in holdout_rows:
        qids.append({"id": r["id"], "doc_type": r["doc_type"], "source": "holdout_ckpt250"})
    for r in external_rows:
        qids.append({"id": r["id"], "doc_type": r["doc_type"], "source": EXTERNAL_SOURCE_TAG})
    OUT_IDS.write_text(json.dumps({"seed": SEED, "count": len(qids), "ids": qids}, ensure_ascii=False, indent=2))
    print(f"wrote {len(qids)} question ids -> {OUT_IDS}")

    # 5. sampling notes
    lines: list[str] = []
    lines.append("# eval100 sampling notes (2026-09-03)\n")
    lines.append("## Composition\n")
    lines.append(f"- Total questions: **100**")
    lines.append(f"- Holdout (ckpt250, genuinely unseen): **{len(holdout_ids)}** (from `holdout_ids_true42.json`)")
    lines.append(f"- External (newly authored, ext_* id namespace): **{len(external_rows)}** (from `external_58_raw.json` -> `external_58.jsonl`)")
    lines.append(f"- Distribution: exactly **4 questions per doc_type × 25 types**.\n")
    lines.append("## External 58 distribution (by backfill vs holdout)\n")
    lines.append("| doc_type | holdout | external | total |")
    lines.append("|---|---|---|---|")
    types = sorted({r["doc_type"] for r in sft_rows})
    holdout_dt = Counter(rows_by_id[i]["doc_type"] for i in holdout_ids)
    external_dt = Counter(r["doc_type"] for r in external_rows)
    for t in types:
        lines.append(f"| {t} | {holdout_dt.get(t,0)} | {external_dt.get(t,0)} | {holdout_dt.get(t,0) + external_dt.get(t,0)} |")
    lines.append("\n## Contamination check\n")
    lines.append(f"- id collisions vs `sft_final_merged.jsonl` (2111 rows): **{len(contam['id_collisions'])}**")
    lines.append(f"- exact-question-text duplicates: **{len(contam['exact_duplicates'])}**")
    lines.append(f"- max SequenceMatcher similarity to any train question: "
                 f"**{max(x['max_sim'] for x in contam['max_similarity_per_question']):.3f}** "
                 f"(threshold 0.5; warnings: {len(contam['warnings'])})")
    lines.append("\nTop-5 most-similar external questions (lower is better, just a sanity check):\n")
    lines.append("| id | doc_type | max_sim | closest training question |")
    lines.append("|---|---|---|---|")
    for x in sorted(contam["max_similarity_per_question"], key=lambda y: -y["max_sim"])[:5]:
        lines.append(f"| {x['id']} | {x['doc_type']} | {x['max_sim']} | {x['closest_train']} |")
    lines.append("\n## Method\n")
    lines.append("External questions are first-person Polish scenarios authored to match the corpus register. "
                 "Each row uses the canonical system prompt and a user message of the new question. "
                 "`tools` is copied from a reference row of the same doc_type (the eval script's "
                 "doc_type enum patch in `eval_holdouts_v2_v3.py` rewrites the enum regardless).")
    lines.append("\nThe merged 100-row file `eval100_questions.jsonl` is the input for the eval script "
                 "(once the script is pointed at it via a new `--input` flag — see eval script TODO).\n")
    OUT_NOTES.write_text("\n".join(lines))
    print(f"wrote sampling notes -> {OUT_NOTES}")

    # summary
    print("\n=== SUMMARY ===")
    print(f"  external_58.jsonl: {len(external_rows)} rows")
    print(f"  eval100_questions.jsonl: {len(eval100)} rows (4/doc_type × 25 types)")
    print(f"  contamination: id_coll={len(contam['id_collisions'])} "
          f"dupes={len(contam['exact_duplicates'])} "
          f"max_sim={max(x['max_sim'] for x in contam['max_similarity_per_question']):.3f}")


if __name__ == "__main__":
    main()
