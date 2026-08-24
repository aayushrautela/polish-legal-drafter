"""Build a kind-balanced umowa_zlecenia anchor set from Modal RAG.

Pipeline (retrieval-only, no LLM):
1. Multi-query semantic sweep across zlecenie KINDS (statute KC 734-751,
   contract clauses, tax/ZUS/min-wage, concrete service domains).
2. Dedup by chunk_id; cap chunks per statute article to force diversity.
3. Kind-balanced greedy selection down to --target anchors.
4. Full-text fetch via chunk_read.
5. Phase-1 style enrichment: umowa_zlecenia template_text + related_clauses
   sampled round-robin from other kinds in the selected pool.

Usage:
  PYTHONPATH=src RETRIEVAL_ENDPOINT=... python build_zlecenie_anchors.py \
      --out-dir outputs/anchors/umowa_zlecenia --target 60
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "src"))

from legal_drafter.retrieval.remote import RemoteRetrieval  # noqa: E402

ART_RE = re.compile(r"art\.?\s*(\d+)", re.IGNORECASE)

QUERIES = {
    "statute": [
        "umowa zlecenia kodeks cywilny art 734 dokonanie czynności prawnej",
        "art 735 wynagrodzenie za zlecenie odpłatność taryfa",
        "art 738 powierzenie wykonania zlecenia osobie trzeciej zastępstwo",
        "art 746 wypowiedzenie zlecenia ważne powody odwołanie",
        "art 744 wynagrodzenie zlecenie po wykonaniu termin zapłaty",
        "art 740 sprawozdanie z wykonania zlecenia informowanie zleceniodawcy",
        "art 742 wydatki zleceniobiorcy zwrot zaliczka zlecenie",
        "art 750 umowa o świadczenie usług odpowiednie stosowanie przepisów o zleceniu",
        "art 491 wypowiedzenie umowy o staranne działanie termin zawód",
        "art 471 nienależyte wykonanie zobowiązania naprawienie szkody",
        "zlecenie śmierć strony art 747 rozwiązanie umowy",
    ],
    "clauses": [
        "umowa zlecenia wynagrodzenie stawka godzinowa postanowienie umowne",
        "ewidencja czasu pracy zleceniobiorca potwierdzenie liczby godzin klauzula",
        "wypowiedzenie umowy zlecenia okres wypowiedzenia postanowienie",
        "zakaz konkurencji poufność klauzula umowa zlecenia",
        "sprawozdanie rozliczenie zlecenia klauzula umowna",
        "zwrot kosztów podróży wydatków umowa zlecenia rozliczanie",
        "oświadczenie zleceniobiorcy inne tytuły do ubezpieczeń społecznych załącznik",
        "rachunek za wykonane usługi termin wystawienia płatność",
        "nieodpłatne zlecenie bez wynagrodzenia umowa",
        "podwykonawca zgoda na powierzenie czynności osobie trzeciej umowa zlecenia",
    ],
    "taxzus": [
        "minimalna stawka godzinowa umowa zlecenia 2026",
        "umowa zlecenia składki ZUS student do 26 roku życia zwolnienie",
        "zbieg tytułów do ubezpieczeń społecznych zlecenie etat minimum",
        "ulga dla młodych PIT umowa zlecenia koszty uzyskania przychodu 20 procent",
        "kontrola PIP minimalna stawka godzinowa kara grzywny zlecenie",
    ],
    "services": [
        "umowa zlecenia sprzątanie mieszkania biura usługi porządkowe",
        "umowa zlecenia marketing social media prowadzenie profilu",
        "umowa zlecenia opieka nad osobą starszą pomoc domowa",
        "umowa zlecenia konsultacje doradztwo usługi termin",
        "umowa zlecenia tłumaczenie tekstów usługa",
        "umowa o świadczenie usług przedsiębiorca usługi naprawa serwis",
    ],
}

KIND_ORDER = ["statute", "clauses", "taxzus", "services"]
QUOTA = {"statute": 24, "clauses": 20, "taxzus": 8, "services": 8}


def classify(text: str, title: str, source_type: str | None) -> str | None:
    t = (text or "").lower()
    # taxzus FIRST (statute chunks about minimum hourly rate / ZUS belong here),
    # then statute, then domain keywords - keyword heuristics must not mislabel
    # statute chunks (e.g. "naprawienie szkody" hitting services "napraw").
    if any(w in t for w in ("stawka godzinowa", "stawki godzinowej", "stawkę godzinową",
                            "stawek godzinowych", "zbieg tytu")) and any(
            w in t for w in ("zlecen", "usług", "świadczeni", "wykonującemu")):
        return "taxzus"
    m = ART_RE.search(t)
    if m and source_type == "statute":
        return "statute"
    if any(w in t for w in ("zus", "składk", "kosztów uzyskania",
                            "podatk", "pit")) and any(
            w in t for w in ("zlecen", "usług", "świadczeni", "wykonującemu")):
        return "taxzus"
    if any(w in t for w in ("sprzątan", "opiekę nad", "opieka nad", "marketing",
                            "social media", "tłumacz", "konsultacj", "doradztw",
                            "serwis", "porządkowe")):
        return "services"
    if "zlecen" in t or "świadczenie usług" in t or "przyjmującym zlecenie" in t:
        return "clauses"
    return None


KC_RE = re.compile(r"du_\d{4}_(\d+)_art_(\d+[a-z]*)")

KC_OK_RANGES = [(353, 357), (361, 363), (385, 386), (471, 472), (476, 486),
                (491, 495), (734, 751)]


def relevant(text: str, chunk_id: str) -> bool:
    """Hard topical gate: must actually concern umowa zlecenia / usługi."""
    t = (text or "").lower()
    if any(w in t for w in ("zleceni", "przyjmującym zlecenie",
                            "dającemu zlecenie", "świadczenie usług")):
        return True
    cid = chunk_id or ""
    # whole acts ingested specifically for zlecenie practice always pass
    if any(p in cid for p in ("du_2024_1773", "du_2025_350", "du_2025_1242",
                              "du_2024_226_art_13", "du_2024_226_art_41",
                              "du_2022_1233_art_11", "du_2022_1233_art_23")):
        return True
    m = KC_RE.search(cid)
    if m and m.group(1) == "1061":
        num = re.match(r"\d+", m.group(2))
        if num:
            try:
                n = int(num.group())
                return any(lo <= n <= hi for lo, hi in KC_OK_RANGES)
            except ValueError:
                return False
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs/anchors/umowa_zlecenia")
    ap.add_argument("--retrieval-url", default=os.environ.get("RETRIEVAL_ENDPOINT"))
    ap.add_argument("--target", type=int, default=60)
    ap.add_argument("--per-article-cap", type=int, default=4)
    ap.add_argument("--related", type=int, default=8)
    args = ap.parse_args()
    if not args.retrieval_url:
        print("ERROR: set RETRIEVAL_ENDPOINT", file=sys.stderr)
        return 2
    remote = RemoteRetrieval(args.retrieval_url)

    cand = {}
    sweep_log = []
    for kind, qs in QUERIES.items():
        for q in qs:
            try:
                res = remote.semantic_search(query=q, top_k=15)
            except Exception as e:
                print(f"[warn] search failed {q[:35]!r}: {e}", file=sys.stderr)
                continue
            items = res if isinstance(res, list) else []
            n_new = 0
            for h in items:
                cid = h.get("chunk_id")
                if not cid:
                    continue
                if cid not in cand:
                    cand[cid] = h
                    n_new += 1
            sweep_log.append({"kind": kind, "query": q, "hits": len(items),
                              "new": n_new})
    print(f"sweep: {len(cand)} unique candidates", flush=True)

    arts_count = Counter()
    by_kind = defaultdict(list)
    skipped_long = 0
    filtered_out = 0
    for cid, h in cand.items():
        text = h.get("text") or ""
        if len(text) < 60:
            continue
        if len(text) > 6000:
            skipped_long += 1
            continue
        if not relevant(text, cid):
            filtered_out += 1
            continue
        k = classify(text, h.get("title") or "", h.get("source_type"))
        if not k:
            continue
        m = ART_RE.search(text)
        art_key = f"{h.get('source_id') or h.get('title')}:{m.group(1)}" if m else cid
        if k == "statute" and arts_count[art_key] >= args.per_article_cap:
            continue
        arts_count[art_key] += 1
        h2 = dict(h)
        h2["_kind"] = k
        h2["_art"] = art_key
        by_kind[k].append(h2)
    print("classified:", {k: len(v) for k, v in by_kind.items()},
          f"(skipped {skipped_long} overlong, {filtered_out} off-topic)", flush=True)

    picked = []
    used_ids = set()
    for k in KIND_ORDER:
        quota = min(QUOTA[k], args.target - len(picked))
        take = by_kind[k][:quota]
        for h in take:
            picked.append(h)
            used_ids.add(h["chunk_id"])

    if len(picked) < args.target:
        fill_order = [k for k in KIND_ORDER if k != "statute"] + ["statute"]
        progress = True
        while len(picked) < args.target and progress:
            progress = False
            for k in fill_order:
                if len(picked) >= args.target:
                    break
                for h in by_kind[k]:
                    if h["chunk_id"] not in used_ids:
                        picked.append(h)
                        used_ids.add(h["chunk_id"])
                        progress = True
                        break
    print(f"picked {len(picked)} / target {args.target}", flush=True)
    if len(picked) < args.target:
        print(f"[warn] only {len(picked)} available; corpus exhausted")

    ids = [h["chunk_id"] for h in picked]
    full = {}
    try:
        for r in remote.chunk_read(chunk_ids=ids, with_adjacent=False) or []:
            if r.get("chunk_id") and r.get("text"):
                full[r["chunk_id"]] = r["text"]
    except Exception as e:
        print(f"[warn] chunk_read failed: {e}", file=sys.stderr)
    for h in picked:
        if len(full.get(h["chunk_id"], "")) > len(h.get("text") or ""):
            h["text"] = full[h["chunk_id"]]

    template_text = ""
    for dt in ("umowa_zlecenia", "umowa_o_swadczenie_uslug"):
        try:
            tpl = remote.get_template(dt)
            txt = (tpl[0].get("text") if isinstance(tpl, list) and tpl else "") or ""
            if txt.strip():
                template_text = txt
                print(f"template via get_template({dt}): {len(txt)} chars", flush=True)
                break
        except Exception as e:
            print(f"[warn] get_template({dt}): {e}", file=sys.stderr)
    if not template_text:
        print("[warn] NO template text resolved - enrichment will be empty",
              file=sys.stderr)

    rr_buckets = [by_kind[k] for k in KIND_ORDER if by_kind.get(k)]
    enriched = []
    for i, h in enumerate(picked):
        rel, seen = [], {h["chunk_id"]}
        rounds = max((len(b) for b in rr_buckets), default=0)
        for j in range(rounds):
            b = rr_buckets[j % len(rr_buckets)]
            if not b:
                continue
            item = b[(i // len(rr_buckets) + j * 7 + i) % len(b)]
            cid = item["chunk_id"]
            if cid in seen:
                continue
            seen.add(cid)
            rel.append(item)
            if len(rel) >= args.related:
                break
        enriched.append({
            "chunk_id": h["chunk_id"],
            "source_ref": h.get("source_ref") or h["chunk_id"],
            "source_type": h.get("source_type"),
            "top_category": h.get("top_category"),
            "title": h.get("title"),
            "display_address": h.get("display_address"),
            "legal_area": h.get("legal_area") or [],
            "doc_types": ["umowa_zlecenia"],
            "doc_type_primary": "umowa_zlecenia",
            "anchor_kind": h["_kind"],
            "template_text": template_text,
            "related_clauses": [
                {"chunk_id": x["chunk_id"],
                 "source_ref": x.get("source_ref") or x.get("chunk_id"),
                 "source_type": x.get("source_type"), "title": x.get("title"),
                 "text": x.get("text")}
                for x in rel[:args.related]
            ],
            "n_related": min(len(rel), args.related),
            "text": h.get("text"),
            "char_len": len(h.get("text") or ""),
        })

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"anchors.zlecenie{len(enriched)}.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in enriched:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    manifest = {
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "doc_type": "umowa_zlecenia",
        "target": args.target,
        "picked": len(enriched),
        "per_article_cap": args.per_article_cap,
        "retrieval_endpoint": args.retrieval_url,
        "queries": sweep_log,
        "kind_distribution": dict(Counter(r["anchor_kind"] for r in enriched)),
        "article_cap_respected": True,
        "output": os.path.basename(out_path),
    }
    with open(os.path.join(out_dir, "build_manifest.json"), "w",
              encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    kc = Counter(r["anchor_kind"] for r in enriched)
    empty_text = sum(1 for r in enriched if not r["text"])
    print(f"DONE. {len(enriched)} anchors -> {out_path}")
    print(f"kinds: {dict(kc)}; empty texts: {empty_text}; "
          f"avg related: {(sum(r['n_related'] for r in enriched)/len(enriched)):.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
