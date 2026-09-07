"""Extract layperson questions from beta-pipeline variation files.

Reads every ``variation_*.json`` under an anchors root
(default: ``outputs/anchors/<doc_type>/variation_*.json``) and flattens the
embedded P3 ``questions`` list into one JSONL row per question::

    {"doc_type": ..., "hint": ..., "question": ..., "source": ..., "variant": ...}

This reproduces the batch question files (``questions.jsonl``,
``questions_ndt.jsonl``, ``questions_5dt.jsonl``, ``questions_b4.jsonl``)
that fed answer synthesis: each is a mechanical flattening of the variation
files that existed at its wave, with ``source`` = ``variation_N.json`` and
``variant`` in {prosta, szczegolowa}. The ``questions_retry.jsonl`` batch
holds re-runs of failed episodes. Output rows are sorted deterministically
by (doc_type, source file, variant) so repeated runs are byte-identical.

Usage:
    python3 scripts/question_gen/extract_questions.py \
        --anchors outputs/anchors --out /tmp/questions_all.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--anchors", default="outputs/anchors",
                    help="anchors root containing <doc_type>/variation_*.json")
    ap.add_argument("--out", required=True, help="output JSONL path")
    args = ap.parse_args(argv)

    rows = []
    files = sorted(glob.glob(os.path.join(args.anchors, "*",
                                          "variation_*.json")))
    for path in files:
        doc_type = os.path.basename(os.path.dirname(path))
        try:
            with open(path, encoding="utf-8") as f:
                var = json.load(f)
        except Exception as e:
            print(f"[skip] {path}: {e!r}", file=sys.stderr)
            continue
        hint = var.get("hint", "")
        for q in var.get("questions", []) or []:
            rows.append({
                "doc_type": doc_type,
                "hint": hint,
                "question": q.get("text", ""),
                "source": os.path.basename(path),
                "variant": q.get("variant", ""),
            })
    rows.sort(key=lambda r: (r["doc_type"], r["source"], r["variant"]))
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[extract] {len(files)} variation files -> {len(rows)} questions "
          f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
