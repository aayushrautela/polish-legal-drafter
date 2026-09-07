"""Fetch Sejm acts whose ELI /struct endpoint is unavailable (textPDF-only).

Downloads the consolidated-text PDF from api.sejm.gov.pl ELI, splits it into
per-article chunks, and emits records with EXACTLY the same schema as
corpus_ingest.fetch_sejm_act so the index builder (scripts/rag/build_index.py)
can embed them unchanged.
origin is set to ``sejm_eli_pdf`` to mark the different extraction path.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import sys
import urllib.request

from pypdf import PdfReader

_UA = {"User-Agent": "polish-legal-drafter/0.1 (+academic research)"}
ACTS = [
    {
        "year": 2025, "position": 350,
        "title": "Ustawa z dnia 13 października 1998 r. o systemie ubezpieczeń społecznych",
        "display": "Dz.U. 2025 poz. 350",
        "out": "dataset/03_drafting_tasks_sources/sources/social_insurance_act_du_2025_350.jsonl",
    },
    {
        "year": 2025, "position": 24,
        "title": "Ustawa z dnia 4 lutego 1994 r. o prawie autorskim i prawach pokrewnych",
        "display": "Dz.U. 2025 poz. 24",
        "out": "dataset/03_drafting_tasks_sources/sources/copyright_act_du_2025_24.jsonl",
    },
    {
        "year": 2025, "position": 1242,
        "title": "Rozporządzenie Rady Ministrów z dnia 11 września 2025 r. w sprawie wysokości minimalnego wynagrodzenia za pracę oraz wysokości minimalnej stawki godzinowej w 2026 r.",
        "display": "Dz.U. 2025 poz. 1242",
        "out": "dataset/03_drafting_tasks_sources/sources/min_wage_rates_reg_du_2025_1242.jsonl",
    },
]
ART_RE = re.compile(r"Art\.\s+(\d+[a-z]*)[\.\s]")


def pdf_text(url: str) -> str:
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = resp.read()
    reader = PdfReader(io.BytesIO(data))
    pages = []
    for pg in reader.pages:
        try:
            pages.append(pg.extract_text() or "")
        except Exception:
            pages.append("")
    raw = "\n".join(pages)
    # de-hyphenate line breaks and squash whitespace
    raw = re.sub(r"-\n(?=[a-ząćęłńóśźż])", "", raw)
    raw = re.sub(r"[ \t]+", " ", raw)
    raw = re.sub(r"\n{2,}", "\n", raw)
    return raw


def split_articles(raw: str) -> list[tuple[str, str]]:
    matches = list(ART_RE.finditer(raw))
    out = []
    for k, m in enumerate(matches):
        end = matches[k + 1].start() if k + 1 < len(matches) else len(raw)
        body = raw[m.end():end]
        body = re.sub(r"\s+", " ", body).strip()
        if len(body) >= 40:
            out.append((m.group(1), f"Art. {m.group(1)}. {body}"))
    return out


def main() -> int:
    total = 0
    for act in ACTS:
        base = f"https://api.sejm.gov.pl/eli/acts/DU/{act['year']}/{act['position']}"
        url = f"{base}/text.pdf"
        print(f"fetching {url}")
        raw = pdf_text(url)
        arts = split_articles(raw)
        if not arts and "§" in raw:
            body = re.sub(r"\s+", " ", raw).strip()
            arts = [("1", body)]
        prefix = f"sejm_eli_{act['publisher'].lower() if False else 'du'}_{act['year']}_{act['position']}"
        eli_url = f"https://eli.gov.pl/eli/DU/{act['year']}/{act['position']}/ogl"
        n = 0
        with open(act["out"], "w", encoding="utf-8") as fh:
            for name, text in arts:
                source_ref = f"sejm_eli:{prefix}_art_{name}"
                source_id = f"{prefix}_art_{name}"
                num = re.match(r"\d+", name)
                rec = {
                    "source_ref": source_ref,
                    "origin": "sejm_eli_pdf",
                    "source_id": source_id,
                    "chunk_id": source_id,
                    "item_id": None,
                    "observation_id": None,
                    "source_type": "statute",
                    "publisher": "Sejm ELI",
                    "title": act["title"],
                    "display_address": act["display"],
                    "article_number": int(num.group()) if num else None,
                    "top_category": None,
                    "risk_or_safe": None,
                    "doc_types": [],
                    "source_url": eli_url,
                    "text": text,
                    "text_hash": hashlib.md5(f"{source_ref}|{text}".encode()).hexdigest()[:16],
                }
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
        total += n
        print(f"  {n} articles -> {act['out']}")
    print(f"TOTAL {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
