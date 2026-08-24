"""Freeze umowa_zlecenia v2: final51 + reviewed admits from the post-ingest
re-sweep (anchors.zlecenie75.jsonl). Nothing from v1 is modified.

New acts' statute chunks are relabeled 'taxzus' (min-wage act du_2024_1773,
social-insurance act du_2025_350) - content-level classification that the
substring matcher got wrong due to Polish inflection.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "anchors.final51.jsonl")
SWEEP = os.path.join(HERE, "anchors.zlecenie75.jsonl")

TAXZUS_IDS = {
    "sejm_eli_du_2024_1773_art_8a",
    "sejm_eli_du_2024_1773_art_8b",
    "sejm_eli_du_2024_1773_art_8c",
    "sejm_eli_du_2024_1773_art_8e",
    "sejm_eli_du_2025_350_art_6",
    "sejm_eli_du_2025_350_art_9",
    "sejm_eli_du_2025_350_art_18",
}
CLAUSE_IDS = {
    "uokik_items:uokik_385338582e7ba050",   # auto-termination on late payment
    "uokik_items:uokik_16ba30e925a46e39",   # mandate ended early -> fee for done work
    "uokik_items:uokik_3bd5fc1c1c11481f",   # extra works priced per cennik
    "uokik_items:uokik_7a60d5c46cffebc9",   # disputes before courts for ZB
}


def main() -> int:
    base = [json.loads(l) for l in open(BASE, encoding="utf-8")]
    assert len(base) == 51
    seen_ids = {r["chunk_id"] for r in base}
    seen_txt = {" ".join((r.get("text") or "").split()) for r in base}

    sweep = {json.loads(l)["chunk_id"]: json.loads(l)
             for l in open(SWEEP, encoding="utf-8")}
    added = []
    for cid in sorted(TAXZUS_IDS | CLAUSE_IDS):
        r = sweep.get(cid)
        if r is None:
            raise SystemExit(f"admit id missing from sweep: {cid}")
        if cid in seen_ids:
            continue
        txt = " ".join((r.get("text") or "").split())
        if txt in seen_txt:
            print(f"[skip exact-text dup] {cid}")
            continue
        if cid in TAXZUS_IDS:
            r["anchor_kind"] = "taxzus"
        seen_ids.add(cid)
        seen_txt.add(txt)
        added.append(r)

    final = base + added
    out_path = os.path.join(HERE, f"anchors.final_v2{len(final)}.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in final:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    payload = json.dumps(final, ensure_ascii=False, sort_keys=True).encode()
    sha = hashlib.sha256(payload).hexdigest()
    with open(os.path.join(HERE, "FROZEN_V2.json"), "w", encoding="utf-8") as f:
        json.dump({"file": os.path.basename(out_path), "sha256": sha,
                   "n_anchors": len(final),
                   "base": os.path.basename(BASE),
                   "frozen_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()},
                  f, indent=2)

    man_path = os.path.join(HERE, "merge_manifest.json")
    man = json.load(open(man_path, encoding="utf-8"))
    man["wave3_v2"] = {
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "base": os.path.basename(BASE),
        "sweep": "anchors.zlecenie75.jsonl (post-ingest re-sweep, target 75)",
        "admitted_taxzus": sorted(TAXZUS_IDS),
        "admitted_clauses": sorted(CLAUSE_IDS),
        "kind_relabel_note": "du_2024_1773 + du_2025_350 statute chunks -> taxzus "
                             "(inflection broke substring classifier: 'stawki godzinowej')",
        "rejected_families": "KC 752-796 agencyjna/komis x6, KK art 148a, KSH art 45, "
                             "sus art 122 repeal-provisions, estate brokerage/courier/"
                             "regulamin/insurance/windykacja clauses, dzielo kara 50000",
        "final_n": len(final),
        "kind_distribution": dict(Counter(r["anchor_kind"] for r in final)),
        "sha256": sha,
    }
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False, indent=2)

    kc = Counter(r["anchor_kind"] for r in final)
    empty = sum(1 for r in final if not r.get("text"))
    print(f"DONE. v2 final={len(final)} -> {out_path}")
    print("kinds:", dict(kc), "| empty:", empty)
    print("sha256:", sha[:16])
    return 0


if __name__ == "__main__":
    sys.exit(main())
