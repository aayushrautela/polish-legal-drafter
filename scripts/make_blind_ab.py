#!/usr/bin/env python3
"""Render clean-42 paired base/LoRA contracts as blind A/B for external judging.

Outputs four artifacts per id:
  - <id>_question.txt   (Polish question + doc_type header)
  - <id>_A.txt          (model A contract body, no metadata)
  - <id>_B.txt          (model B contract body, no metadata)
  - <id>_manifest.json  (only for the operator: which is base, which is LoRA)

The mapping (A↔base, B↔LoRA) is randomized per id with a fixed seed (7)
so the operator must look at the manifest to learn which is which, and
can re-randomize by changing the seed.

Usage:
  python scripts/make_blind_ab.py
"""
from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "manual_reads_blind"
OUT.mkdir(parents=True, exist_ok=True)

BASE_PATH = ROOT / "outputs" / "eval100" / "base_true42.jsonl"
LORA_PATH = ROOT / "outputs" / "eval100" / "lora250_true42.jsonl"
QUESTIONS = ROOT / "outputs" / "qa_pairs_all_final" / "sft_final_merged.jsonl"

# IDs we already analyzed.  Add more here as we go.
IDS = [1011, 1971, 1534, 904]
SEED = 7


def extract_contract_body(rec: dict) -> str:
    """Best-effort extraction of the contract string from a scenario record."""
    c = rec.get("contract") or rec.get("draft") or rec.get("raw_draft_response", {}).get("content", "")
    m = re.search(r"```json\s*(\{.*)", c, re.S)
    body = c
    if m:
        try:
            body = json.loads(re.sub(r"```\s*$", "", m.group(1)).strip())["contract"]
        except Exception:
            # fall back to slicing from the first §
            m2 = re.search(r"(UMOWA|§\s*1)", c)
            if m2:
                body = c[m2.start():]
    return body.replace("\\n", "\n").strip()


def main() -> None:
    base = {json.loads(l)["id"]: json.loads(l) for l in open(BASE_PATH)}
    lora = {json.loads(l)["id"]: json.loads(l) for l in open(LORA_PATH)}
    qs = {json.loads(l)["id"]: json.loads(l) for l in open(QUESTIONS)}

    rng = random.Random(SEED)
    summary = []
    for qid in IDS:
        if qid not in base or qid not in lora or qid not in qs:
            print(f"skip {qid}: missing in source files")
            continue
        a_rec, b_rec = base[qid], lora[qid]
        # randomize letter assignment
        order = ["A", "B"]
        rng.shuffle(order)
        a_letter, b_letter = order  # which letter gets base vs lora
        # build mapping {A: 'base'|'lora', B: the other}
        mapping = {a_letter: "base", b_letter: "lora"}
        records = {"base": a_rec, "lora": b_rec}
        for letter in ("A", "B"):
            (OUT / f"{qid}_{letter}.txt").write_text(extract_contract_body(records[mapping[letter]]) + "\n")
        # question file
        q = qs[qid]
        (OUT / f"{qid}_question.txt").write_text(
            f"doc_type: {q['doc_type']}\nid: {qid}\n\n{q['question'].strip()}\n"
        )
        manifest = {
            "id": qid,
            "doc_type": q["doc_type"],
            "seed": SEED,
            "mapping": mapping,  # e.g. {"A": "lora", "B": "base"}
        }
        (OUT / f"{qid}_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
        summary.append((qid, q["doc_type"], mapping))

    print(f"wrote {len(summary)} blind pairs to {OUT}")
    for qid, dt, m in summary:
        print(f"  {qid} ({dt}): A={m['A']:>5}  B={m['B']:>5}")


if __name__ == "__main__":
    main()
