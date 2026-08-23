"""Phase 1: enrich each anchor with the real RAG clauses it can mutate.

For every anchor we attach:
* ``template_text`` - the real CC-BY-4.0 template for its doc_type (the base set
  of clauses/conditions the LLM may REMOVE from).
* ``related_clauses`` - a small bank of OTHER real RAG chunks (same doc_type, then
  same legal_area) the LLM may ADD as new conditions. These are already-fetched
  verbatim RAG chunks from the Phase-0 pool, so every add/remove stays grounded.

Retrieval-only (Modal): 12 get_template calls; no LLM.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

from legal_drafter.retrieval.remote import RemoteRetrieval
from legal_drafter.retrieval.templates import DOC_TYPE_QUERY


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="outputs/anchors/anchor_pool.jsonl")
    ap.add_argument("--out", default="outputs/anchors/anchor_pool.phase1.jsonl")
    ap.add_argument("--retrieval-url", default=os.environ.get("RETRIEVAL_ENDPOINT"))
    ap.add_argument("--related", type=int, default=8)
    args = ap.parse_args()
    if not args.retrieval_url:
        print("ERROR: set RETRIEVAL_ENDPOINT or --retrieval-url", file=sys.stderr)
        return 2
    remote = RemoteRetrieval(args.retrieval_url)

    recs = [json.loads(l) for l in open(args.inp)]

    templates = {}
    for dt in DOC_TYPE_QUERY:
        try:
            t = remote.get_template(dt)
            templates[dt] = (t[0].get("text") if t else "") or ""
        except Exception as e:
            print(f"[warn] get_template {dt}: {e}", file=sys.stderr)
            templates[dt] = ""

    by_dt = defaultdict(list)
    by_la = defaultdict(list)
    for r in recs:
        for d in r.get("doc_types") or []:
            by_dt[d].append(r)
        for la in r.get("legal_area") or []:
            by_la[la].append(r)

    for r in recs:
        dts = r.get("doc_types") or []
        dt = dts[0] if dts else None
        rel = [x for x in by_dt.get(dt, []) if x["chunk_id"] != r["chunk_id"]] if dt else []
        if len(rel) < args.related:
            for la in (r.get("legal_area") or []):
                for x in by_la.get(la, []):
                    if x["chunk_id"] != r["chunk_id"] and x not in rel:
                        rel.append(x)
                if len(rel) >= args.related:
                    break
        rel = rel[: args.related]
        r["doc_type_primary"] = dt
        r["template_text"] = templates.get(dt) if dt else None
        r["related_clauses"] = [
            {
                "chunk_id": x["chunk_id"],
                "source_ref": x.get("source_ref"),
                "source_type": x.get("source_type"),
                "title": x.get("title"),
                "text": x.get("text"),
            }
            for x in rel
        ]
        r["n_related"] = len(rel)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with_tmpl = sum(1 for r in recs if r.get("template_text"))
    print(f"DONE. {len(recs)} anchors enriched -> {args.out}")
    print(f"with template_text: {with_tmpl}; avg related_clauses: "
          f"{sum(r['n_related'] for r in recs)/max(1,len(recs)):.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
