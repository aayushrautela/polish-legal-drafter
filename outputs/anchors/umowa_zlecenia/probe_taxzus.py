"""Probe corpus for taxzus + replacement content for umowa_zlecenia set.

Read-only diagnostic: runs semantic_search AND keyword_search against the
GPU retrieval endpoint and dumps every unique hit with a snippet so the
results can be READ before anything is added to the frozen set.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "src"))

from legal_drafter.retrieval.remote import RemoteRetrieval  # noqa: E402

SEM_QUERIES = [
    "minimalna stawka godzinowa umowa zlecenia",
    "ewidencjonowanie czasu pracy zleceniobiorca potwierdzenie godzin",
    "stawka godzinowa wypłata wynagrodzenia raz w miesiącu",
    "kara grzywny minimalna stawka godzinowa inspekcja pracy",
    "zbieg tytułów ubezpieczeń społecznych zlecenie etat",
    "składki ZUS student ucznia umowa zlecenia zwolnienie",
    "ulga dla młodych do 26 roku życia zwolnienie podatku",
    "koszty uzyskania przychodu 20 procent umowa zlecenia ryczałt",
    "ryczałt od przychodów ewidencjonowanych usługi stawka",
    "oświadczenie zleceniobiorcy inne tytuły ubezpieczeń",
    "zakaz konkurencji poufność umowa zlecenia klauzula",
    "wypowiedzenie umowy zlecenia okres klauzula umowna",
]
KW_QUERIES = [
    "stawka godzinowa",
    "zleceniobiorca składki",
    "ewidencja czasu pracy",
    "ulga dla młodych",
]

def main() -> int:
    url = os.environ.get("RETRIEVAL_ENDPOINT")
    remote = RemoteRetrieval(url)

    seen = {}
    def add(h, how):
        cid = h.get("chunk_id")
        if not cid or cid in seen:
            return
        h["_how"] = how
        seen[cid] = h

    for q in SEM_QUERIES:
        try:
            for h in remote.semantic_search(query=q, top_k=15) or []:
                add(h, f"sem:{q[:38]}")
        except Exception as e:
            print(f"[warn] sem {q[:30]}: {e}", file=sys.stderr)
    for q in KW_QUERIES:
        try:
            for h in remote.keyword_search(query=q, top_k=25) or []:
                add(h, f"kw:{q}")
        except Exception as e:
            print(f"[warn] kw {q}: {e}", file=sys.stderr)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe_taxzus.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for h in seen.values():
            f.write(json.dumps(h, ensure_ascii=False) + "\n")
    print(f"unique hits: {len(seen)} -> {out}")

    # compact readable listing grouped by rough bucket
    for cid, h in seen.items():
        t = (h.get("text") or "").replace("\n", " ")
        addr = h.get("display_address") or ""
        print(f"\n== {cid}\n   {addr} | src={h.get('source_type')} | via={h['_how']}")
        print(f"   {t[:220]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
