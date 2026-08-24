"""v2 top-up after full-corpus zlecenie audit: final_v262 -> final_v265.

Adds 3 hand-read statute chunks found by the unused-content audit:
 - min-wage act art. 8d (EXCEPTIONS to 8a-8c: commission-only + full freedom)
 - social-insurance act art. 8 (pracownik definition incl. zlecenie with own
   employer -> ZUS/reclassification tie-in)
 - KC art. 751 (2-year limitation for zlecenie payment claims)
FROZEN_V2.json is updated to point at the new file; final_v262 stays on disk.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "anchors.final_v262.jsonl")
ADD = [
    ("sejm_eli_du_2024_1773_art_8d", "taxzus",
     "dataset/03_drafting_tasks_sources/sources/min_wage_act_du_2024_1773.jsonl"),
    ("sejm_eli_du_2025_350_art_8", "taxzus",
     "dataset/03_drafting_tasks_sources/sources/social_insurance_act_du_2025_350.jsonl"),
    ("sejm_eli_du_2024_1061_art_751", "statute",
     "../../dataset/03_drafting_tasks_sources/sources/cleaned_rag_chunks.jsonl"),
]


def main() -> int:
    import sys
    sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "src"))
    from legal_drafter.retrieval.remote import RemoteRetrieval

    base = [json.loads(l) for l in open(BASE, encoding="utf-8")]
    assert len(base) == 62
    seen_ids = {r["chunk_id"] for r in base}
    import re as _re
    def _norm(t):
        return _re.sub(r"^Obwieszczenie Marszałka Sejmu Rzeczypospolitej Polskiej.*?(?=Art\.|§|Ustawa|ROZPORZ)", "", " ".join((t or "").split()), flags=_re.S)
    seen_txt = {_norm(r.get("text")) for r in base}
    tpl_text = next((r.get("template_text") for r in base if r.get("template_text")), "")

    added = []
    remote = RemoteRetrieval(os.environ["RETRIEVAL_ENDPOINT"])
    for cid, kind, local_path in ADD:
        text = ""
        src_ref = cid
        display = ""
        title = ""
        if local_path and os.path.exists(local_path):
            for l in open(local_path, encoding="utf-8"):
                r = json.loads(l)
                if r.get("chunk_id") == cid:
                    text = r.get("text") or ""
                    src_ref = r.get("source_ref") or cid
                    display = r.get("display_address") or ""
                    title = r.get("title") or ""
                    break
        else:
            rows = remote.chunk_read(chunk_ids=[cid], with_adjacent=False) or []
            row = next((x for x in rows if x.get("chunk_id") == cid), {}) or {}
            text = row.get("text") or ""
            display = row.get("display_address") or ""
            title = row.get("title") or ""
            src_ref = row.get("source_ref") or cid
        assert len(text) > 60, f"empty/short text for {cid}"
        import re as _re
        norm = _re.sub(r"^Obwieszczenie Marszałka Sejmu Rzeczypospolitej Polskiej.*?(?=Art\.|§|Ustawa|ROZPORZ)", "", " ".join(text.split()), flags=_re.S)
        assert norm not in seen_txt, f"text dup {cid}"
        seen_txt.add(norm)
        pool = [r for r in base if r["chunk_id"] != cid]
        step = max(1, len(pool) // 8)
        rel = [pool[(k * step) % len(pool)] for k in range(8)]
        added.append({
            "chunk_id": cid,
            "source_ref": src_ref,
            "source_type": "statute",
            "top_category": None,
            "title": title,
            "display_address": display,
            "legal_area": [],
            "doc_types": ["umowa_zlecenia"],
            "doc_type_primary": "umowa_zlecenia",
            "anchor_kind": kind,
            "template_text": tpl_text,
            "related_clauses": [
                {"chunk_id": x["chunk_id"], "source_ref": x.get("source_ref"),
                 "source_type": x.get("source_type"), "title": x.get("title"),
                 "text": x.get("text")}
                for x in rel[:8]
            ],
            "n_related": min(len(rel), 8),
            "text": text,
            "char_len": len(text),
        })

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
    man["wave4_v2_audit_topup"] = {
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "trigger": "user check: unused zleceni* content across ALL corpus files "
                   "(285 raw hits; mostly junk families + id-less cleaned_rag_chunks "
                   "text-dups of kept clauses)",
        "admitted": [{"chunk_id": c, "kind": k} for c, k, _ in ADD],
        "notable_rejects": [
            "sus art. 6a (childcare benefit titles), sus art. 16 (generic rate catalog)",
            "uokik bezzwrotna oplat clause (credit-capital tail pollution)",
            "uokik kancelaria sub-delegation (windykacja family consistency)",
        ],
        "final_n": len(final),
        "kind_distribution": dict(Counter(r["anchor_kind"] for r in final)),
        "sha256": sha,
    }
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False, indent=2)

    print(f"DONE. v2 final={len(final)} -> {out_path}")
    print("kinds:", dict(Counter(r["anchor_kind"] for r in final)))
    print("sha256:", sha[:16])
    return 0


if __name__ == "__main__":
    main()
