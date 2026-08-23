"""Phase 0: build the RAG-anchored instruction anchor pool (hybrid, Modal-only).

Anchors are REAL legal_corpus chunks (statute articles / clauses / rulings) that
(1) are tied to a doc_type via the real CC-BY-4.0 templates (A: template-anchored)
and (2) extend into legal areas the 12 templates don't cover (B: breadth).
No local Qdrant is opened; every lookup goes through the Modal retrieval service.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sys
from collections import Counter

from legal_drafter.retrieval.remote import RemoteRetrieval
from legal_drafter.retrieval.templates import DOC_TYPE_QUERY

ART_RE = re.compile(r"art\.?\s*(\d+)", re.IGNORECASE)

BREADTH_QUERIES = [
    "wady rzeczy sprzedanej rękojmia",
    "odstąpienie od umowy przez konsumenta",
    "zwrot zaliczki i kaucji",
    "podnajem lokalu",
    "gwarancja sprzedawcy",
    "odszkodowanie za nienależyte wykonanie zobowiązania",
    "odsetki ustawowe za opóźnienie",
    "przedawnienie roszczeń",
    "wynagrodzenie za bezpodstawne wzbogacenie",
    "rozwiązanie umowy ze skutkiem natychmiastowym",
    "zadatek a zaliczka",
    "odpowiedzialność solidarna poręczyciela",
    "pełnomocnictwo do czynności prawnych",
    "klauzula niedozwolona rejestr UOKiK",
    "kara umowna art 483 kodeksu cywilnego",
]


def parse_arts(text):
    if not text:
        return []
    out = []
    seen = set()
    for m in ART_RE.finditer(text):
        a = m.group(1)
        if a not in seen:
            seen.add(a)
            out.append(a)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/anchors/anchor_pool.jsonl")
    ap.add_argument("--retrieval-url", default=os.environ.get("RETRIEVAL_ENDPOINT"))
    ap.add_argument("--k", type=int, default=12)
    args = ap.parse_args()
    if not args.retrieval_url:
        print("ERROR: set RETRIEVAL_ENDPOINT or --retrieval-url", file=sys.stderr)
        return 2
    remote = RemoteRetrieval(args.retrieval_url)

    anchors = {}

    def add_hit(hit, dt=None, kind="A_template", arts=None):
        cid = hit.get("chunk_id") or hit.get("source_ref")
        if not cid:
            return
        if cid in anchors:
            rec = anchors[cid]
            if dt and dt not in rec["doc_types"]:
                rec["doc_types"].append(dt)
            if arts:
                for a in arts:
                    if a not in rec["cited_arts"]:
                        rec["cited_arts"].append(a)
            rec["kinds"].add(kind)
            return
        anchors[cid] = {
            "chunk_id": cid,
            "source_ref": hit.get("source_ref") or cid,
            "source_type": hit.get("source_type"),
            "top_category": hit.get("top_category"),
            "title": hit.get("title"),
            "display_address": hit.get("display_address"),
            "legal_area": hit.get("legal_area") or [],
            "doc_types": [dt] if dt else [],
            "cited_arts": list(arts or []),
            "kinds": {kind},
        }

    for dt, query in DOC_TYPE_QUERY.items():
        tmpl = remote.get_template(dt)
        text = (tmpl[0].get("text") if tmpl else "") or ""
        arts = parse_arts(text)
        q = query
        if arts:
            q = f"{query} art {' '.join('art.' + a for a in arts[:6])}"
        try:
            hits = remote.semantic_search(query=q, top_k=args.k)
        except Exception as e:
            print(f"[warn] semantic_search failed for {dt}: {e}", file=sys.stderr)
            hits = []
        for h in hits or []:
            add_hit(h, dt=dt, kind="A_template", arts=arts)
        if arts:
            try:
                kh = remote.keyword_search(
                    keywords=[f"art. {a}" for a in arts[:8]], top_k=args.k
                )
            except Exception as e:
                print(f"[warn] keyword_search failed for {dt}: {e}", file=sys.stderr)
                kh = []
            for h in kh or []:
                add_hit(h, dt=dt, kind="A_template", arts=arts)

    for bq in BREADTH_QUERIES:
        try:
            hits = remote.semantic_search(query=bq, top_k=args.k)
        except Exception as e:
            print(f"[warn] breadth semantic_search failed for {bq!r}: {e}", file=sys.stderr)
            hits = []
        for h in hits or []:
            add_hit(h, dt=None, kind="B_breadth")

    ids = list(anchors.keys())
    full_texts = {}
    if ids:
        try:
            res = remote.chunk_read(chunk_ids=ids, with_adjacent=False)
            for r in res or []:
                cid = r.get("chunk_id")
                if cid and r.get("text"):
                    full_texts[cid] = r.get("text")
        except Exception as e:
            print(f"[warn] chunk_read failed: {e}", file=sys.stderr)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    n = 0
    with open(args.out, "w") as f:
        for cid, rec in anchors.items():
            ks = rec.pop("kinds")
            rec["anchor_kind"] = (
                "A+B"
                if {"A_template", "B_breadth"} <= ks
                else ("A_template" if "A_template" in ks else "B_breadth")
            )
            rec["text"] = full_texts.get(cid)
            rec["anchor_id"] = f"anc_{re.sub(r'[^0-9a-zA-Z]', '_', str(cid))[:60]}"
            rec["char_len"] = len(rec["text"] or "")
            rec["n_doc_types"] = len(rec["doc_types"])
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1

    kind_c = Counter(r["anchor_kind"] for r in anchors.values())
    dt_c = Counter()
    for r in anchors.values():
        for d in r["doc_types"]:
            dt_c[d] += 1
    print(f"DONE. {n} unique anchors -> {args.out}")
    print("by kind:", dict(kind_c))
    print("A matches per doc_type:", dict(dt_c))
    return 0


if __name__ == "__main__":
    sys.exit(main())
