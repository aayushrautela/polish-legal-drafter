"""Targeted repair of the umowa_zlecenia candidate set after manual review.

Reads anchors.zlecenie60.jsonl, drops reviewed-junk entries (reasons logged),
reclassifies mislabeled statute chunks, adds hand-verified replacements
(PIT art. 41, UZNK arts 11/23, three uokik clauses fetched via chunk_read),
exact-text dedupes, and writes anchors.finalNN.jsonl + FROZEN.json +
merge_manifest.json.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "src"))

from legal_drafter.retrieval.remote import RemoteRetrieval  # noqa: E402

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "anchors.zlecenie60.jsonl")

DROPS = {
    6: "statute out of range: KC art 759 (umowa agencyjna chapter)",
    9: "statute out of range: KC art 763 (agencyjna)",
    10: "statute out of range: KC art 764(1) (agencyjna)",
    13: "statute out of range: KC art 760(2) (agencyjna)",
    17: "statute out of range: KC art 796 (poza zleceniem)",
    18: "statute out of range: KC art 752 (komis chapter)",
    54: "wrong statute: Kodeks karny art 148a (substring trap 'zleceni')",
    58: "generic KC art 70(5) - not zlecenie-specific",
    32: "exact duplicate of index 28",
    43: "exact duplicate of index 34",
    52: "exact duplicate of index 51",
    26: "wrong domain: budowlany Inwestor/Wykonawca wstrzymanie zaplaty",
    27: "wrong domain: budowlany Zamawiający odstapienie robót",
    29: "wrong domain: pośrednictwo nieruchomości wynagrodzenie",
    31: "wrong domain: ubezpieczeniowy odszkodowanie Zleceniobiorcy",
    34: "wrong domain: pośrednictwo nieruchomości umowa przedwstępna",
    37: "wrong domain: kurierski Nadawca/odbiorca cesja wierzytelności",
    40: "wrong domain: ubezpieczenia postępowanie likwidacyjne",
    41: "wrong domain: pośrednictwo nieruchomości zakaz sąsiedzki",
    42: "wrong domain: sprzedaż Sprzedający/Kupujący kara umowna zadatek",
    51: "wrong domain: regulamin sklepu cesja praw",
    53: "off-type: umowa do dochodzenia roszczeń (adwokacka) bez świadczenia usług",
}
RECLASSIFY = {i: "statute" for i in (45, 46, 47, 48, 49, 50)}

ADD_IDS = [
    ("sejm_eli_du_2024_226_art_41", "taxzus"),
    ("sejm_eli_du_2022_1233_art_11", "clauses"),
    ("sejm_eli_du_2022_1233_art_23", "clauses"),
    ("uokik_items:uokik_bed886a11d4e34e3", "clauses"),
    ("uokik_items:uokik_e18b1879a00e0d0a", "clauses"),
    ("uokik_items:uokik_7b0febcda0adcd7a", "clauses"),
]


def main() -> int:
    recs = [json.loads(l) for l in open(SRC, encoding="utf-8")]
    assert len(recs) == 60, f"expected 60 rows, got {len(recs)}"

    kept = []
    for i, r in enumerate(recs):
        if i in DROPS:
            continue
        if i in RECLASSIFY:
            r["anchor_kind"] = RECLASSIFY[i]
        kept.append(r)

    # exact-text dedupe safety net
    seen_txt, deduped = set(), []
    for r in kept:
        key = " ".join((r.get("text") or "").split())
        if key in seen_txt:
            continue
        seen_txt.add(key)
        deduped.append(r)
    kept = deduped

    remote = RemoteRetrieval(os.environ["RETRIEVAL_ENDPOINT"])
    probe = {}
    pl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe_taxzus.jsonl")
    for l in open(pl_path, encoding="utf-8"):
        h = json.loads(l)
        probe[h.get("chunk_id")] = h

    tpl_text = next((r.get("template_text") for r in kept if r.get("template_text")), "")
    added = []
    for cid, kind in ADD_IDS:
        hit = probe.get(cid) or {}
        rows = remote.chunk_read(chunk_ids=[cid], with_adjacent=False) or []
        full = next((r.get("text") for r in rows if r.get("chunk_id") == cid), None) \
            or hit.get("text") or ""
        rel_srcs = [r for r in kept if r["chunk_id"] != cid]
        step = max(1, len(rel_srcs) // 8)
        rel = [rel_srcs[(k * step) % len(rel_srcs)] for k in range(8)]
        added.append({
            "chunk_id": cid,
            "source_ref": hit.get("source_ref") or cid,
            "source_type": hit.get("source_type"),
            "top_category": hit.get("top_category"),
            "title": hit.get("title"),
            "display_address": hit.get("display_address"),
            "legal_area": hit.get("legal_area") or [],
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
            "text": full,
            "char_len": len(full),
        })

    final = kept + added

    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, f"anchors.final{len(final)}.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in final:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    payload = json.dumps(final, ensure_ascii=False, sort_keys=True).encode()
    sha = hashlib.sha256(payload).hexdigest()
    with open(os.path.join(out_dir, "FROZEN.json"), "w", encoding="utf-8") as f:
        json.dump({"file": os.path.basename(out_path), "sha256": sha,
                   "n_anchors": len(final),
                   "frozen_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()},
                  f, indent=2)

    from collections import Counter
    manifest = {
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "base": os.path.basename(SRC),
        "dropped": [{"index": i, "chunk_id": recs[i]["chunk_id"], "reason": why}
                    for i, why in sorted(DROPS.items())],
        "reclassified_to_statute": sorted(RECLASSIFY.keys()),
        "added": [{"chunk_id": c, "kind": k} for c, k in ADD_IDS],
        "final_n": len(final),
        "kind_distribution": dict(Counter(r["anchor_kind"] for r in final)),
        "sha256": sha,
    }
    with open(os.path.join(out_dir, "merge_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"DONE. final={len(final)} -> {out_path}")
    print("kinds:", dict(Counter(r['anchor_kind'] for r in final)))
    print("sha256:", sha[:16])
    return 0


if __name__ == "__main__":
    sys.exit(main())
