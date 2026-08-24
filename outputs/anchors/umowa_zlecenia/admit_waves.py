"""Wave admission for umowa_zlecenia expansion: final44 -> final51.

Only hand-read chunks are admitted (see merge_manifest.json wave2 section).
Everything here was reviewed by reading full/snippet text before inclusion;
all other gathered candidates were explicitly rejected (junk domains,
wrong statutes, amortization articles, abusive-pattern specimens).
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "src"))

from legal_drafter.retrieval.remote import RemoteRetrieval  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "anchors.final44.jsonl")

ADMIT = [
    ("sejm_eli_du_2024_226_art_13", "taxzus",
     "PIT art.13 - przychody z dzialalnosci wykonywanej osobiscie; pkt 8 "
     "definiuje przychody z umow zlecenia/o dzielo (web-verified MF source)"),
    ("sejm_eli_du_2024_1061_art_363", "statute",
     "KC art.363 - forma naprawienia szkody (przywrócenie stanu vs suma "
     "pieniezna); grounds zlecenie liability/damages clauses"),
    ("extractor_v9:broad_0021_curated_eval_label_ebb0955bf35ceae0", "clauses",
     "curated principle: kara umowna za samo opóznienie w zapłacie "
     "wynagrodzenia niezaleznie od odsetek"),
    ("extractor_v9:broad_0025_curated_eval_label_a9f9913d29071827", "clauses",
     "curated principle: po rezygnacji zamawiajacego rozliczenie wykonanych "
     "prac i rzeczywistych kosztów, zwrot niewykorzystanej zaliczki"),
    ("uokik_items:uokik_f257b46580e0aaf9", "clauses",
     "clause: prawo wstrzymania uslug przy nieuregulowaniu rachunku w terminie"),
    ("uokik_items:uokik_032b42b1d236cf1b", "clauses",
     "clause: platne wezwanie do zaplaty po nieoplaconej fakturze"),
    ("uokik_items:uokik_1176ceb45c030028", "clauses",
     "clause: start przedmiotu umowy warunkowany dostarczeniem dokumentów "
     "i wniesieniem oplaty"),
]


def main() -> int:
    recs = [json.loads(l) for l in open(SRC, encoding="utf-8")]
    assert len(recs) == 44
    existing = {r["chunk_id"] for r in recs}
    tpl_text = next((r.get("template_text") for r in recs if r.get("template_text")), "")

    probe = {}
    for l in open(os.path.join(HERE, "probe_taxzus.jsonl"), encoding="utf-8"):
        h = json.loads(l)
        probe[h.get("chunk_id")] = h
    waves = {}
    for l in open(os.path.join(HERE, "wave_candidates.jsonl"), encoding="utf-8"):
        h = json.loads(l)
        waves[h["chunk_id"]] = h

    remote = RemoteRetrieval(os.environ["RETRIEVAL_ENDPOINT"])
    added = []
    for cid, kind, why in ADMIT:
        if cid in existing:
            print(f"[skip already-present] {cid}")
            continue
        src = waves.get(cid) or probe.get(cid) or {"chunk_id": cid, "text": ""}
        rows = remote.chunk_read(chunk_ids=[cid], with_adjacent=False) or []
        full = next((r.get("text") for r in rows if r.get("chunk_id") == cid), None) \
            or src.get("text") or ""
        assert len(full) > 60, f"empty text for {cid}"
        pool = [r for r in recs if r["chunk_id"] != cid]
        step = max(1, len(pool) // 8)
        rel = [pool[(k * step) % len(pool)] for k in range(8)]
        added.append({
            "chunk_id": cid,
            "source_ref": src.get("source_ref") or cid,
            "source_type": src.get("source_type"),
            "top_category": src.get("top_category"),
            "title": src.get("title"),
            "display_address": src.get("display_address"),
            "legal_area": src.get("legal_area") or [],
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

    final = recs + added
    out_path = os.path.join(HERE, f"anchors.final{len(final)}.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in final:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    payload = json.dumps(final, ensure_ascii=False, sort_keys=True).encode()
    sha = hashlib.sha256(payload).hexdigest()
    with open(os.path.join(HERE, "FROZEN.json"), "w", encoding="utf-8") as f:
        json.dump({"file": os.path.basename(out_path), "sha256": sha,
                   "n_anchors": len(final),
                   "frozen_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()},
                  f, indent=2)

    man_path = os.path.join(HERE, "merge_manifest.json")
    man = json.load(open(man_path, encoding="utf-8"))
    man["wave2_expansion"] = {
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "base": "anchors.final44.jsonl",
        "gathered_candidates": 171,
        "reviewed": "top-scored subset incl. all wave1/wave2/wave4 + scored wave3; "
                    "full-text or snippet READ before every admit",
        "admitted": [{"chunk_id": c, "kind": k, "why": w} for c, k, w in ADMIT],
        "notable_rejects": [
            "PIT arts 22a/22d/22e/22f/22h/22l/22o/23a = fixed-asset AMORTIZATION "
            "and leasing definitions, not zlecenie taxation",
            "KP chunk was art. 22^3 (e-mail monitoring), not §1 reclassification test; "
            "corpus lacks KP 22 §1 itself",
            "sejm_eli_du_2024_1568_art_736 = KPC security-measure procedure, not KC 736",
            "KC arts 741/749/750^1/353^1/385^1/484^2 absent from corpus under tried ids",
            "banking payment-order 'zlecenie' clauses, estate brokerage, studies/TOS, "
            "travel-operator, abusive-pattern specimen fragments",
        ],
        "final_n": len(final),
        "sha256": sha,
    }
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False, indent=2)

    from collections import Counter
    kc = Counter(r["anchor_kind"] for r in final)
    print(f"DONE. final={len(final)} -> {out_path}")
    print("kinds:", dict(kc))
    print("sha256:", sha[:16])
    return 0


if __name__ == "__main__":
    sys.exit(main())
