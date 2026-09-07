"""Merge batch question files into one question set + manifest.

Concatenates the given batch JSONL files in order with no deduplication:
rows are kept verbatim, including the ``questions_retry.jsonl`` re-runs,
matching the historical merge. Each row is stamped with a 1-based ``id``
(restarted per input file) and ``source_file`` (input basename), reproducing
the historical schema. A manifest recording per-file counts is written,
mirroring ``outputs/merged_question_set/manifest.json``:

    {"created_at": ..., "source_files": {name: n}, "total_questions": ...,
     "total_doc_types": ..., "per_doc_type": {...}}

Historical merge (2026-08-29): questions.jsonl (864) + questions_ndt.jsonl
(248) + questions_5dt.jsonl (398) + questions_b4.jsonl (550) +
questions_retry.jsonl (44) = 2,104 rows -> questions_merged.jsonl.

Usage:
    python3 scripts/question_gen/merge_questions.py \
        --in a.jsonl b.jsonl --out questions_merged.jsonl \
        --manifest manifest.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
from collections import Counter


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="inputs", nargs="+", required=True,
                    help="batch JSONL files in merge order")
    ap.add_argument("--out", required=True, help="merged output JSONL path")
    ap.add_argument("--manifest", default=None,
                    help="manifest JSON path (default: <out dir>/manifest.json)")
    args = ap.parse_args(argv)

    rows = []
    source_files = {}
    for path in args.inputs:
        n = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                n += 1
                r = dict(r)
                r.pop("id", None)
                r.pop("source_file", None)
                r["id"] = str(n)
                r["source_file"] = os.path.basename(path)
                rows.append(r)
        source_files[os.path.basename(path)] = n
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    per_doc = Counter(r.get("doc_type", "") for r in rows)
    manifest = {
        "created_at": datetime.date.today().isoformat(),
        "source_files": source_files,
        "total_questions": len(rows),
        "total_doc_types": len(per_doc),
        "per_doc_type": dict(sorted(per_doc.items())),
    }
    mpath = args.manifest or os.path.join(os.path.dirname(args.out) or ".",
                                          "manifest.json")
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[merge] {len(args.inputs)} files -> {len(rows)} rows -> "
          f"{args.out} (+ {mpath})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
