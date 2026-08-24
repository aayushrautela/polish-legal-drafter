"""GATHER-only pass for umowa_zlecenia expansion (waves 1-4). No admission here.

Wave 1: full texts of the 17 phase1-pool anchors missing from final44 + hunt
        for PIT art. 13 pkt 8.
Wave 2: UOKiK abusive-clause hits already stored in probe_taxzus.jsonl.
Wave 3: fresh semantic+keyword query angles.
Wave 4: direct chunk_read probes for unpicked in-range KC statute articles.

Writes wave_candidates.jsonl (everything, tagged _wave) and prints a readable
listing for manual review. Admission happens ONLY after human reads it.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "src"))

from legal_drafter.retrieval.remote import RemoteRetrieval  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FINAL44 = {json.loads(l)["chunk_id"]
           for l in open(os.path.join(HERE, "anchors.final44.jsonl"), encoding="utf-8")}

W1_IDS = [
    "sejm_eli_du_2024_226_art_13",
    "sejm_eli_du_2023_1465_art_22",
    "sejm_eli_du_2024_226_art_22a", "sejm_eli_du_2024_226_art_22d",
    "sejm_eli_du_2024_226_art_22e", "sejm_eli_du_2024_226_art_22f",
    "sejm_eli_du_2024_226_art_22h", "sejm_eli_du_2024_226_art_22l",
    "sejm_eli_du_2024_226_art_22o", "sejm_eli_du_2024_226_art_23a",
    "sejm_eli_du_2024_1796_art_16", "sejm_eli_du_2024_37_art_217",
    "uokik_items:uokik_b911a389e7f56a50", "uokik_items:uokik_8fe6c7b047ed12c5",
    "uokik_items:uokik_4906e13cea90ee6f", "uokik_items:uokik_9df747432ec060c8",
    "uokik_items:uokik_6e3ab360de951a3c",
    "extractor_v9:broad_0021_curated_eval_label_ebb0955bf35ceae0",
]

W4_IDS = ["sejm_eli_du_2024_1061_art_736", "sejm_eli_du_2024_1061_art_741",
          "sejm_eli_du_2024_1061_art_749", "sejm_eli_du_2024_1061_art_7501",
          "sejm_eli_du_2024_1061_art_75011", "sejm_eli_du_2024_1061_art_3531",
          "sejm_eli_du_2024_1061_art_3851", "sejm_eli_du_2024_1061_art_4842"]

W3_SEM = [
    "umowa o świadczenie usług ochrony osób i mienia monitoring",
    "umowa zlecenia prowadzenie ksiąg rachunkowych księgowość",
    "umowa zlecenia usługi sprzątania wspólnota mieszkaniowa",
    "rachunek termin wystawienia płatność wynagrodzenia zlecenie klauzula",
    "zwrot kosztów podróży delegacji zleceniobiorcy rozliczenie",
    "materiały powierzone zużycie rozliczenie wykonawca umowa",
    "zakaz konkurencji przez określony czas po zakończeniu umowy",
    "ryczałt ewidencjonowany stawka 8,5 procent 12 15 usługi budowlane",
    "wynagrodzenie ryczałtowe zlecenie kwota słownie klauzula",
    "zaliczka na podatek dochodowy pobierana przez płatnika zlecenie",
    "odstępne kara za rezygnację z usługi rezerwacji terminu",
    "art 736 kodeks cywilny zbieg prawa do wynagrodzenia i zwrotu wydatków",
    "art 741 kodeks cywilny naprawienie szkody zleceniobiorcą",
    "art 749 kodeks cywilny obowiązek zapłaty bez wezwania",
    "art 3531 kodeks cywilny treść stosunku zobowiązaniowego",
]
W3_KW = ["zleceniobiorcy przysługuje", "wynagrodzenie ryczałtowe w wysokości",
         "nie będzie świadczyć usług na rzecz"]


def main() -> int:
    remote = RemoteRetrieval(os.environ["RETRIEVAL_ENDPOINT"])
    out = []

    def add(h, wave, how):
        cid = h.get("chunk_id")
        if not cid or cid in FINAL44:
            return False
        for x in out:
            if x["chunk_id"] == cid:
                return False
        h["_wave"] = wave
        h["_how"] = how
        out.append(h)
        return True

    # wave 1 + 4: direct reads
    for wave, ids in ((1, W1_IDS), (4, W4_IDS)):
        rows = remote.chunk_read(chunk_ids=ids, with_adjacent=False) or []
        got = {}
        for r in rows:
            if r.get("chunk_id") and r.get("text"):
                got[r["chunk_id"]] = r
        for cid in ids:
            h = dict(got.get(cid) or {"chunk_id": cid, "text": ""})
            add(h, wave, "chunk_read")
            if cid not in got:
                print(f"[miss] w{wave} {cid}")

    # wave 2: abusive-clause hits already in probe file
    n2 = 0
    for l in open(os.path.join(HERE, "probe_taxzus.jsonl"), encoding="utf-8"):
        h = json.loads(l)
        if str(h.get("chunk_id", "")).startswith("uokik_abusive_clause"):
            n2 += add(h, 2, "probe:abusive")
    print(f"[w2] abusive-clause candidates: {n2}")

    # wave 3: fresh queries
    for q in W3_SEM:
        try:
            for h in remote.semantic_search(query=q, top_k=12) or []:
                add(dict(h), 3, f"sem:{q[:36]}")
        except Exception as e:
            print(f"[warn] {q[:30]}: {e}", file=sys.stderr)
    for q in W3_KW:
        try:
            for h in remote.keyword_search(query=q, top_k=20) or []:
                add(dict(h), 3, f"kw:{q}")
        except Exception as e:
            print(f"[warn] kw {q}: {e}", file=sys.stderr)

    with open(os.path.join(HERE, "wave_candidates.jsonl"), "w", encoding="utf-8") as f:
        for h in out:
            f.write(json.dumps(h, ensure_ascii=False) + "\n")

    print(f"\nTOTAL candidates: {len(out)}")
    for i, h in enumerate(out):
        t = " ".join((h.get("text") or "").split())
        addr = (h.get("display_address") or "")[:40]
        print(f"\n[{i:03d}] w{h['_wave']} {h['chunk_id'][:58]} | {addr}")
        print(f"      {t[:200]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
