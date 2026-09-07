#!/usr/bin/env python3
"""Tally blind pairwise judge annotations into pair verdicts.

Implements the finalized swap+repeat rule (PLAN_EVAL100 methodology):

Per (judge, order) cell: trials must be unanimous for a stable
order-verdict (K=2, so majority == 2/2; a 1-1 split is an unstable order).
Per judge: both order-verdicts must agree -> consistent judge verdict.
  - stable orders pointing at different winners = position flip (bias).
  - any unstable order = noise (flag, never a silent win).
Per pair: both judges consistent AND agreeing -> WIN (or clean TIE).
  Anything else -> TIE with a flag (position_flip / unstable_order /
  cross_judge_disagree). Only consistent wins count; this is the
  Zheng/MT-Bench both-orderings rule extended to repeats (Soumik 2026:
  repeats separate bias from noise so clear-cut 8/8 wins survive).

Diagnostics per set (Shi 2025 / Yagubyan 2026): swap_consistent,
repetition_stable (fraction of unanimous judge-order cells),
cross_judge_agree.

Usage:
  PYTHONPATH=src python3 scripts/tally_judge.py --annotations <path> [--out <path>]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def tally_set(votes: list[dict]) -> dict:
    """Tally one (question, pair) set. Each vote needs judge, order,
    trial, verdict_mapped. Returns the verdict record."""
    by_cell: dict[tuple, list] = defaultdict(list)
    for v in votes:
        by_cell[(v["judge"], v["order"])].append(v["verdict_mapped"])

    order_verdict: dict[tuple, str | None] = {}
    for cell, vals in by_cell.items():
        vals = [x for x in vals if x is not None]
        order_verdict[cell] = vals[0] if vals and all(x == vals[0] for x in vals) else None

    judges = sorted({v["judge"] for v in votes})
    judge_verdict: dict[str, str | None] = {}
    flags: set[str] = set()
    for j in judges:
        f, r = order_verdict.get((j, "fwd")), order_verdict.get((j, "rev"))
        if f is not None and f == r:
            judge_verdict[j] = f
        elif f is None or r is None:
            judge_verdict[j] = None
            flags.add("unstable_order")
        elif f != r and f != "tie" and r != "tie":
            judge_verdict[j] = None
            flags.add("position_flip")
        else:
            judge_verdict[j] = None
            flags.add("order_disagree")

    decided = [jv for jv in judge_verdict.values() if jv is not None]
    total = len(votes)
    if len(decided) == len(judges) and len(set(decided)) == 1:
        winner = decided[0]
        strength = sum(1 for v in votes if v["verdict_mapped"] == winner) / total
        if winner == "tie":
            verdict, confidence = "tie", "medium"  # consistent tie
        elif strength >= 1.0:
            verdict, confidence = winner, "high"
        elif strength >= 0.75:
            verdict, confidence = winner, "medium"
        else:
            verdict, confidence = winner, "low"
    else:
        verdict, confidence = "tie", "low"
        strength = 0.0
        if len(decided) == len(judges):
            flags.add("cross_judge_disagree")

    rep_cells = len(by_cell)
    rep_stable = sum(1 for cell in by_cell
                     if order_verdict[cell] is not None) / rep_cells if rep_cells else 0.0
    return {
        "votes": total,
        "order_verdicts": {f"{j}/{o}": v for (j, o), v in sorted(order_verdict.items())},
        "judge_verdicts": judge_verdict,
        "verdict": verdict,
        "strength": round(strength, 3),
        "confidence": confidence,
        "flags": sorted(flags),
        "swap_consistent": all(jv is not None for jv in judge_verdict.values())
        and len(set(judge_verdict.values())) == 1,
        "repetition_stable": round(rep_stable, 3),
        "cross_judge_agree": len(set(judge_verdict.values())) == 1,
        "vote_counts": dict(Counter(v["verdict_mapped"] for v in votes)),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    sets: dict[tuple, list] = defaultdict(list)
    errors = 0
    with open(args.annotations, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("status") != "ok":
                errors += 1
                continue
            sets[(str(r["question_id"]), r["pair"])].append(r)

    results = []
    for (qid, pair), votes in sorted(sets.items()):
        rec = {"question_id": qid, "pair": pair, "complete": len(votes) == 8}
        rec.update(tally_set(votes))
        results.append(rec)

    complete = [r for r in results if r["complete"]]
    wins = Counter((r["pair"], r["verdict"]) for r in complete)
    summary = {
        "rows": sum(len(v) for v in sets.values()) + errors,
        "error_rows": errors,
        "sets": len(results),
        "complete_sets": len(complete),
        "pair_verdicts": {f"{p}|{v}": c for (p, v), c in sorted(wins.items())},
        "flags": dict(Counter(f for r in complete for f in r["flags"])),
    }
    out = {"summary": summary, "sets": results}
    text = json.dumps(out, ensure_ascii=False, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        print(text)
    print(f"sets={len(results)} complete={len(complete)} "
          f"verdicts={summary['pair_verdicts']} flags={summary['flags']}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
