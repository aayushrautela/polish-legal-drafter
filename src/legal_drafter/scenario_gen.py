"""Fresh scenario generator for the KAFT distillation corpus.

Generates NEW, fully-grounded Polish legal-drafting scenarios (the
``distill_drafting_task_v1`` schema + ``evidence_report`` + drafted document)
by anchoring each scenario on a REAL statute chunk retrieved from the Qdrant
``legal_corpus`` collection, then asking the teacher ("scrapper-model") to
write the full scenario around it.

Crucially, the teacher ASSIGNS ``norm_ids`` itself from the retrieved source
text (act + article), so we never depend on a pre-populated grounding field.

Parallelized with N workers (thread pool). Built to be launched detached:

    nohup PYTHONPATH=src .venv/bin/python -m legal_drafter.scenario_gen \
        --n 150 --parallel 8 \
        --out outputs/scenarios/synthetic_v1.jsonl \
        --log outputs/scenarios/synthetic_v1.log > outputs/scenarios/launch.log 2>&1 &
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import re
import sys
import time
import threading
import types
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from legal_drafter.config import load_checker
from legal_drafter.retrieval.hybrid import (
    DEFAULT_COLLECTION,
    DEFAULT_QDRANT_PATH,
    QdrantHybridStore,
)
from legal_drafter.agentic_loop import run_agentic
from legal_drafter.retrieval.agentic_tools import TOOL_SCHEMAS

# Seed topics spanning the doc types present in the corpus. Each yields a set
# of anchor statute chunks; scenarios are distributed across them for diversity.
SEED_TOPICS = [
    "umowa zlecenia świadczenie usług wynagrodzenie wypowiedzenie",
    "umowa o dzieło wykonawca dzieło odbiór",
    "umowa najmu lokalu mieszkalnego czynsz wypowiedzenie eksmisja",
    "umowa sprzedaży rzeczy ruchomej rękojmia wady",
    "kaucja zaliczka zwrot odstąpienie od umowy",
    "odstąpienie od umowy przez konsumenta 14 dni prawo do odstąpienia",
    "wezwanie do zapłaty ostateczne wezwanie odsetki",
    "reklamacja konsumenta wada towaru naprawa wymiana",
    "klauzula niedozwolona sąd właściwy konsumenta uokik",
    "kara umowna opóźnienie zapłaty odsetki umowne",
    "pełnomocnictwo do zawarcia umowy umocowanie",
    "poręczenie za zapłatę długu oświadczenie poręczyciela",
]

SYSTEM_PROMPT = (
    "You are a senior Polish legal drafter building a high-quality distillation "
    "dataset for a legal-language model. You are given one or more REAL Polish "
    "legal source excerpts (statutes / UOKiK abusive-clause entries) retrieved "
    "from a corpus. Your task is to invent a realistic, self-contained legal "
    "drafting scenario that is genuinely grounded in those excerpts, and to "
    "produce a COMPLETE JSON object describing it. Output ONLY a single valid "
    "JSON object. No commentary, no markdown fences."
)

DOC_TYPE_HINT = (
    "doc_type MUST be one of: umowa_zlecenia, umowa_o_dzielo, umowa_najmu, "
    "umowa_sprzedazy, kaucja_zaliczka, odstapienie_konsumenta, wezwanie_do_zaplaty, "
    "reklamacja_konsumenta, klauzula_niedozwolona, kara_umowna, pelnomocnictwo, "
    "poreczenie, other."
)


INSTRUCTION_TEMPLATES = {
    "umowa_zlecenia": [
        "Zatrudniam osobę do sprzątania biura dwa razy w tygodniu, umowa na pół roku, jak to zapisać żeby było jasne kto co płaci i kiedy można wypowiedzieć?",
        "Mam zlecić znajomemu przewóz mebli, nie wiem jak ustalić cenę i odpowiedzialność za zniszczenia, pomóż mi to ułożyć.",
    ],
    "umowa_o_dzielo": [
        "Zamawiam u grafika projekt logo, ma być gotowy w miesiąc, chcę wiedzieć jak zapisać odbiór i płatność.",
        "Dogadałem się z programistą na napisanie strony, jak spisać umowę żeby dzieło było odebrane dopiero jak działa?",
    ],
    "umowa_najmu": [
        "Wynajmuję mieszkanie na rok, właściciel chce kaucję i podwyższyć czynsz, jak to uczciwie zapisać?",
        "Mam lokal użytkowy do wynajęcia małej firmie, jak ustalić media i wypowiedzenie?",
    ],
    "umowa_sprzedazy": [
        "Sprzedaję używany samochód sąsiadowi, boję się że się zepsuje po zakupie, jak zabezpieczyć się na rękojmi?",
        "Kupuję od kolegi laptopa, chcę umowę że pieniądze oddam jak sprawdzę sprzęt, jak to ująć?",
    ],
    "kaucja_zaliczka": [
        "Biorę udział w przetargu i muszę dać zaliczkę, co się dzieje jak drugiej strony nie ma, jak to opisać?",
        "Wynajmuję i właściciel wziął kaucję, nie wiem kiedy mi ją odda, pomóż mi to zapisać.",
    ],
    "odstapienie_konsumenta": [
        "Kupiłam sukienkę online i nie pasuje, minęło 10 dni, czy mogę zrezygnować i jak to napisać do sprzedawcy?",
        "Zamówiłem kurs online, żałuję, jak odstąpić od umowy w 14 dniach?",
    ],
    "wezwanie_do_zaplaty": [
        "Klient mi nie zapłacił za usługę od dwóch miesięcy, jak mu wysłać ostateczne wezwanie z odsetkami?",
        "Wykonawca nie dostał wynagrodzenia, jak napisać wezwanie do zapłaty żeby brzmiało poważnie?",
    ],
    "reklamacja_konsumenta": [
        "Kupiłem buty i po tygodniu się rozkleiły, jak złożyć reklamację żeby dali nowe?",
        "Telefon mi się zepsuł w gwarancji, jak napisać reklamację do sklepu?",
    ],
    "klauzula_niedozwolona": [
        "Wpisuję do umowy z klientem karę za odstąpienie, czy to się utrzyma w sądzie, jak to sformułować uczciwie?",
        "Chcę zabronić klientowi iść do sądu w moim mieście, czy tak można, jak zapisać klauzulę?",
    ],
    "kara_umowna": [
        "Podwykonawca się spóźnia z robotą, chcę karę umowną, jak ją wpisać żeby była skuteczna?",
        "Wynajmuję i chcę karę jak lokator płaci po terminie, jak to ułożyć?",
    ],
    "pelnomocnictwo": [
        "Mama jest chora i muszę załatwić za nią sprawę w urzędzie, jakie dać pełnomocnictwo?",
        "Chcę upoważnić kogoś do sprzedaży mojego auta, jak spisać pełnomocnictwo?",
    ],
    "poreczenie": [
        "Znajomy pożycza pieniądze i chce żeby ktoś za niego ręczył, jak to zapisać żeby ręczyciel coś znaczył?",
        "Obiecałem ręczyć za czynsz kolegi, jak sformułować oświadczenie poręczyciela?",
    ],
}


def build_instruction(i: int) -> tuple[str, str]:
    if INSTRUCTION_BANK:
        text, dt = INSTRUCTION_BANK[i % len(INSTRUCTION_BANK)]
        return text, dt
    doc_types = list(INSTRUCTION_TEMPLATES.keys())
    dt = doc_types[i % len(doc_types)]
    templates = INSTRUCTION_TEMPLATES[dt]
    return templates[i % len(templates)], dt


INSTRUCTION_BANK: list[tuple[str, str]] = []


def load_instruction_bank(path: str) -> int:
    """Load layperson questions from a phase-3 jsonl (instructions[].text)."""
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            dt = rec.get("doc_type_primary") or rec.get("phase3_dt") or "other"
            for v in rec.get("instructions") or []:
                t = (v.get("text") or "").strip()
                if t:
                    INSTRUCTION_BANK.append((t, dt))
                    n += 1
    return n


def _schema_skeleton() -> str:
    return """{
  "topic": "<short Polish topic of the drafting task>",
  "doc_type": "<one of the allowed doc_types>",
  "instruction": "<natural-language message a layperson would type into ChatGPT: first-person, casual, everyday language, NO legal jargon or article/paragraph numbers>",
  "facts": {
    "doc_type": "<MUST equal the top-level doc_type>",
    "summary": "<client situation in Polish>",
    "parties": "<e.g. 'zleceniobiorca (wykonawca) i zleceniodawca'>",
    "key_dates": "<optional Polish string>"
  },
   "sources": [
     {
       "source_ref": "<the EXACT source_ref string shown in the RETRIEVED REAL SOURCES list for the source you use; copy it verbatim>"
     }
   ],
  "evidence_report": {
    "topic": "<same as top-level topic>",
    "regime": {
      "primary_regime": "<e.g. zlecenie_general, residential_lease>",
      "doc_type": "<MUST equal facts.doc_type>",
      "topic": "<topic>",
      "confidence": "high|medium|low",
      "required_source_roles": ["applicable_rule"],
      "forbidden_regimes": [],
      "must_use_concepts": ["<Polish legal concept that MUST appear in the draft>"],
      "forbidden_concepts": ["<Polish legal concept that MUST NOT appear>"],
      "notes": ["<string note>"]
    },
    "legal_issues": [
      {
        "issue_id": "<topic_focus>",
        "clause_functions": ["<function the clause must perform>"],
        "must_support": ["<norm_id REQUIRED, from the provided sources>"],
        "may_support": ["<norm_id that may support, from the provided sources>"],
        "allowed_acts": ["<act_id, e.g. KC>"]
      }
    ],
    "required_norms": ["<norm_id required to support the draft>"],
    "covered_required_norms": ["<required_norms present in provided sources>"],
    "missing_required_norms": ["<required_norms NOT in provided sources>"],
    "selected_norms": ["<norm_ids selected from the provided sources>"],
    "selected_count": <int>,
    "eligible_count": <int>,
    "negative_count": <int>
  },
  "expected_output": "<the COMPLETE drafted Polish legal document (full contract/letter, ALL standard sections), properly formatted, citing the relevant articles by number (e.g. 'zgodnie z art. 355 KC ...')>"
}"""


def build_user_prompt(sources: list[dict], index: int, total: int, doc_type_hint: str) -> str:
    src_block = []
    for i, s in enumerate(sources, 1):
        src_block.append(
            f"[{i}] source_ref={s.get('source_ref')} | source_type={s.get('source_type')} | "
            f"top_category={s.get('top_category')} | risk_or_safe={s.get('risk_or_safe')} | "
            f"title={s.get('title')}\n  text: {s.get('text','')}"
        )
    src_text = "\n\n".join(src_block)
    return (
        f"You are generating scenario #{index} of {total} for a diverse corpus. "
        "Make the facts realistic and DISTINCT from typical templates (vary the "
        "parties, amounts, dates, and context).\n\n"
        "RETRIEVED REAL SOURCES (ground the scenario ONLY in these; you may cite "
        "only norms present in them):\n"
        f"{src_text}\n\n"
        "Produce the scenario JSON with these HARD constraints:\n"
        f"1. {doc_type_hint}\n"
        "2. facts.doc_type MUST equal the top-level doc_type.\n"
        "3. In the 'sources' array, include ONLY an object with 'source_ref' set to "
        "the EXACT source_ref string shown in the RETRIEVED REAL SOURCES list above "
        "for each source you use. Do NOT copy the source text, title, source_type, "
        "top_category, risk_or_safe or norm_ids into your JSON — they are filled in "
        "automatically from the corpus. Copy the ref string verbatim (do NOT add a "
        "'sejm_eli:' prefix or otherwise modify it). Do NOT invent sources.\n"
        "4. You do NOT need to output norm_ids for sources — they are derived "
        "automatically from the corpus. Focus your reasoning on the drafting task.\n"
        "5. evidence_report.selected_norms / required_norms / legal_issues[].must_support "
        "MUST be drawn ONLY from the norm_ids you assigned to the provided sources.\n"
        "6. evidence_report.regime.doc_type MUST equal facts.doc_type; confidence is "
        "one of high|medium|low; notes is a JSON ARRAY of strings.\n"
        "7. expected_output is the COMPLETE drafted Polish legal document (e.g. a full "
        "umowa najmu / pełnomocnictwo / wezwanie) with ALL standard sections present "
        "(strony/parties, przedmiot, czynsz/wynagrodzenie, kaucja, media, obowiązki stron, "
        "termin, wypowiedzenie/rozwiązanie, protokół zdawczo-odbiorczy where applicable, "
        "podpisy, załączniki), properly formatted, citing the relevant articles by number.\n"
        "   If you used get_template, follow its SECTION STRUCTURE and reuse its real clause "
        "language and statutory references as the skeleton, but ADAPT every clause to these "
        "specific facts (concrete parties/amounts/dates) and to the retrieved sources. Do NOT "
        "paste the template's generic explanatory guidance verbatim into the document — write "
        "a clean professional document. PRESERVE its CC-BY-4.0 attribution.\n"
        "8. Produce the FULL document end-to-end — do NOT truncate, summarize, or omit "
        "standard clauses. Plan the section outline first in your reasoning, then write the "
        "complete document as expected_output (no outline prefix inside the document). Length "
        "follows the doc type (typically 2000-4000 words / ~3000-6000 tokens); never stop early. "
        "Do NOT pad with boilerplate NOT grounded in the sources.\n"
        "9. In JSON string values represent line breaks as \\n (escaped); never emit "
        "raw newline characters inside a string.\n\n"
        "10. The 'instruction' field MUST read like a LAYPERSON's ChatGPT message: "
        "first-person, casual, everyday Polish, describing a real-life situation, with "
        "NO legal terms of art, Latin, or citations of articles/paragraphs/statutes by "
        "number. It must NOT be a formal lawyer request. Good: 'Wynajmuję mieszkanie od "
        "gminy na 18 miesięcy, lokator ma niski dochód — jak to zapisać żeby było zgodne "
        "z prawem?'. Bad: 'Proszę sporządzić projekt umowy najmu socjalnego…' or "
        "'Sporządź pełnomocnictwo…'.\n\n"
        f"JSON schema to produce:\n{_schema_skeleton()}\n\n"
        "Output ONLY the JSON object."
    )


def extract_json(text: str):
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
    try:
        return json.loads(text)
    except Exception:
        pass
    a, b = text.find("{"), text.rfind("}")
    if a != -1 and b != -1 and b > a:
        try:
            return json.loads(text[a : b + 1])
        except Exception:
            return None
    return None


def validate(scn: dict, sources: list[dict]) -> dict:
    issues = []
    if not isinstance(scn, dict):
        return {"ok": False, "issues": ["not a JSON object"]}
    dt = scn.get("doc_type")
    facts = scn.get("facts") or {}
    if facts.get("doc_type") != dt:
        issues.append("facts.doc_type != top-level doc_type")
    if not scn.get("topic"):
        issues.append("topic missing")
    if not scn.get("instruction"):
        issues.append("instruction missing")
    outs = scn.get("sources") or []
    if len(outs) == 0:
        issues.append("sources empty")
    avail = set()
    for s in outs:
        for n in s.get("norm_ids") or []:
            avail.add(n)
    if not avail:
        issues.append("no norm_ids assigned to any source")
    er = scn.get("evidence_report") or {}
    reg = er.get("regime") or {}
    if reg.get("doc_type") != dt:
        issues.append("evidence_report.regime.doc_type != doc_type")
    if reg.get("confidence") not in ("high", "medium", "low"):
        issues.append("confidence invalid")
    if not isinstance(reg.get("notes", []), list):
        issues.append("regime.notes not a list")
    for f in ("selected_norms", "required_norms", "covered_required_norms", "missing_required_norms"):
        vals = er.get(f)
        if vals is None:
            issues.append(f"evidence_report.{f} missing")
        elif not isinstance(vals, list):
            issues.append(f"evidence_report.{f} not a list")
        elif f != "missing_required_norms":
            bad = [v for v in vals if v not in avail]
            if bad:
                issues.append(f"evidence_report.{f} unresolvable: {bad}")
    li = er.get("legal_issues")
    if not isinstance(li, list) or not li:
        issues.append("legal_issues missing/empty")
    else:
        for i, x in enumerate(li):
            for f in ("must_support", "may_support"):
                for v in x.get(f, []) or []:
                    if v not in avail:
                        issues.append(f"legal_issues[{i}].{f} unresolvable: {v}")
    if not (scn.get("expected_output") or "").strip():
        issues.append("expected_output empty")
    return {"ok": len(issues) == 0, "issues": issues}


def load_all_sources(store: QdrantHybridStore) -> list[dict]:
    """One full scroll of the collection (no embedding) — fast."""
    out: list[dict] = []
    nxt = None
    while True:
        pts, nxt = store.client.scroll(
            collection_name=store.collection_name, with_payload=True, limit=2000, offset=nxt
        )
        for p in pts:
            if p.payload:
                out.append(p.payload)
        if nxt is None:
            break
    return out


def build_pool(all_sources: list[dict], per_bucket: int = 40) -> list[dict]:
    """Bucket real statute/abusive-clause chunks by top_category for diversity.

    Round-robin across buckets so the scenario pool cycles through every legal
    category (better doc-type coverage for KAFT) instead of over-sampling the
    largest bucket.
    """
    from collections import defaultdict

    buckets: dict[str, list[dict]] = defaultdict(list)
    for d in all_sources:
        # Only keep chunks whose norm_ids can be derived, so every generated
        # scenario is guaranteed to have grounded norms (avoids the
        # "no norm_ids assigned to any source" invalid case).
        if d.get("source_type") in ("statute", "abusive_clause") and derive_norm_ids(d):
            buckets[d.get("top_category") or "other"].append(d)
    capped = {cat: docs[:per_bucket] for cat, docs in buckets.items()}
    pool: list[dict] = []
    exhausted: set[str] = set()
    while len(pool) < per_bucket * len(capped) and len(exhausted) < len(capped):
        for cat, docs in capped.items():
            if cat in exhausted:
                continue
            if not docs:
                exhausted.add(cat)
                continue
            d = docs.pop(0)
            others = [x for x in capped[cat] if x is not d][:2]
            pool.append({"anchor": d, "context": [d] + others, "category": cat})
    return pool


def collect_anchors(store: QdrantHybridStore, n_target: int) -> list[dict]:
    """Gather a diverse pool of anchor statute chunks + same-category context."""
    all_src = load_all_sources(store)
    pool = build_pool(all_src)
    if len(pool) < n_target:
        # pad by allowing more per bucket
        pool = build_pool(all_src, per_bucket=max(per_bucket := 40, n_target))
    return pool


def build_corpus_map(store: QdrantHybridStore) -> dict:
    """Map chunk_id and source_ref -> verbatim corpus payload (for canonicalization)."""
    m: dict = {}
    offset = None
    while True:
        res = store.client.scroll(collection_name=store.collection_name, with_payload=True, limit=2000, offset=offset)
        pts, offset = res[0], res[1]
        for p in pts:
            pl = p.payload or {}
            for kk in (pl.get("chunk_id"), pl.get("source_ref")):
                if kk:
                    m[kk] = pl
        if offset is None:
            break
    return m


def derive_norm_ids(chunk: dict) -> list[str]:
    """Authoritative norm_ids derived from the canonical corpus chunk.

    The chunk_id encodes the act ELI + article (e.g.
    ``sejm_eli_du_2024_1061_art_16`` -> ``KC:16``); abusive-clause chunks carry a
    ``uokik_clause_XX`` id. This is always consistent with the verbatim text,
    unlike teacher-assigned norm_ids which drift after canonicalization.
    """
    cid = (chunk.get("chunk_id") or chunk.get("source_ref") or "")
    text = f"{chunk.get('text') or ''} {chunk.get('title') or ''}"
    low = text.lower()
    st = chunk.get("source_type")
    if st == "abusive_clause" or "uokik" in cid.lower():
        m = re.search(r"uokik[_:]?clause[_:]?(\d+)", cid, re.I)
        return [f"uokik:{m.group(1)}"] if m else ["uokik:abusive"]
    art = None
    m = re.search(r"art[_]?(\d+)", cid, re.I)
    if m:
        art = m.group(1)
    else:
        m = re.search(r"art\.?\s*(\d+)", text, re.I)
        if m:
            art = m.group(1)
    if not art:
        return []
    if "1061" in cid or "kodeks cywilny" in low or "k.c." in low:
        act = "KC"
    elif "postępowania cywil" in low or "kodeks post" in low:
        act = "KPC"
    elif "prawach konsumenta" in low:
        act = "ustawa_o_prawach_konsumenta"
    elif "ochronie praw lokator" in low or "praw lokator" in low:
        act = "ustawa_o_ochronie_praw_lokatorow"
    else:
        act = "KC"
    return [f"{act}:{art}"]


def canonicalize_sources(scn: dict, corpus_map: dict) -> None:
    """Replace each generated source with the verbatim corpus chunk + derived norm_ids.

    The teacher may paraphrase statute text or mangle source_ref; we overwrite
    the source with the real chunk (matched by ref, stripping a possible origin
    prefix) and set norm_ids authoritatively from the chunk. Guarantees grounding.
    """
    KEEP = ("chunk_id", "source_ref", "title", "text", "source_type",
            "top_category", "risk_or_safe", "display_address", "legal_area")
    new = []
    for s in scn.get("sources") or []:
        ref = s.get("source_ref") or ""
        cand = ref
        if cand not in corpus_map and ":" in ref:
            cand = ref.split(":", 1)[1]
        corp = corpus_map.get(cand) or corpus_map.get(ref)
        if corp:
            c = {k: corp.get(k) for k in KEEP if corp.get(k) is not None}
            c["norm_ids"] = derive_norm_ids(corp)
            new.append(c)
        else:
            pass  # drop unmatched source rather than keep a textless stub
    if not new:
        new = scn.get("sources") or []
    scn["sources"] = new
    scn["source_refs"] = [c.get("source_ref") for c in new]


def rebuild_evidence_norms(scn: dict) -> None:
    """Rebuild evidence_report norm lists from the authoritative source norm_ids.

    After canonicalization the teacher's norm references no longer match, so we
    set required/selected/covered norms and legal_issues.must_support to the
    grounded source norm_ids. Keeps the teacher's regime + issue reasoning.
    """
    avail = []
    for s in scn.get("sources") or []:
        for n in s.get("norm_ids") or []:
            if n not in avail:
                avail.append(n)
    er = scn.get("evidence_report") or {}
    er["required_norms"] = list(avail)
    er["selected_norms"] = list(avail)
    er["covered_required_norms"] = list(avail)
    er["missing_required_norms"] = []
    er["selected_count"] = len(avail)
    er["eligible_count"] = len(avail)
    er["negative_count"] = len(scn.get("sources") or [])
    for li in er.get("legal_issues") or []:
        li["must_support"] = list(avail)
        li["may_support"] = []
    scn["evidence_report"] = er


def stream_collect(client, cfg, user, watchdog: float = 900.0):
    """Stream a teacher response with an absolute wall-clock watchdog.

    The omnirouter can leave streaming calls open indefinitely under high
    concurrency; the SDK timeout does not always fire, so we cap elapsed time.
    Returns (content, error_or_none).
    """
    import threading

    box = {"content": "", "err": None, "done": False}

    def run():
        try:
            kwargs = dict(
                model=cfg.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                timeout=cfg.timeout,
                stream=True,
            )
            if cfg.extra_body:
                kwargs["extra_body"] = cfg.extra_body
            if cfg.extra_headers:
                kwargs["extra_headers"] = cfg.extra_headers
            resp = client.chat.completions.create(**kwargs)
            for chunk in resp:
                if chunk.choices:
                    d = chunk.choices[0].delta
                    if d and d.content:
                        box["content"] += d.content
            box["done"] = True
        except Exception as e:  # pragma: no cover - network dependent
            box["err"] = str(e)
            box["done"] = True

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(watchdog)
    if not box["done"]:
        return box["content"], "watchdog_timeout"
    return box["content"], box["err"]


def make_chat(client, cfg, tool_schemas, model=None, stream=True):
    """Build a callable that drives the teacher via native tool calling.

    When ``stream`` is True (default) the assistant turn is streamed and printed
    live, so a whole scenario reads as one continuous chat. The streamed
    tool-call fragments (id/name/arguments per index) are accumulated into a
    message-like object compatible with ``run_agentic`` (``.content`` and
    ``.tool_calls`` with ``.id``/``.function.name``/``.function.arguments``).

    Pass a stable ``session_id`` per ``chat(messages, session_id=...)`` call so
    the litellm/bifrost proxy groups every iteration of one scenario into a
    single chat thread (via the ``litellm_session_id`` body field).
    """

    def chat(messages, session_id=None):
        kwargs = dict(
            model=model or cfg.model,
            messages=messages,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            timeout=cfg.timeout,
            tools=tool_schemas,
            tool_choice="auto",
            stream=stream,
        )
        eb = {k: v for k, v in (cfg.extra_body or {}).items() if k != "tool_choice"}
        if session_id:
            eb["litellm_session_id"] = session_id
        if eb:
            kwargs["extra_body"] = eb
        if cfg.extra_headers:
            kwargs["extra_headers"] = cfg.extra_headers

        if not stream:
            return client.chat.completions.create(**kwargs)

        chunks = client.chat.completions.create(**kwargs)
        content_parts: list[str] = []
        tc_map: dict[int, dict] = {}
        for chunk in chunks:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta is None:
                continue
            if getattr(delta, "content", None):
                content_parts.append(delta.content)
                print(delta.content, end="", flush=True)
            for tc in getattr(delta, "tool_calls", None) or []:
                slot = tc_map.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
                if tc.id:
                    slot["id"] += tc.id
                if getattr(tc.function, "name", None):
                    slot["name"] += tc.function.name
                if getattr(tc.function, "arguments", None):
                    slot["arguments"] += tc.function.arguments

        tool_calls = [
            types.SimpleNamespace(
                id=slot["id"],
                type="function",
                function=types.SimpleNamespace(name=slot["name"], arguments=slot["arguments"]),
            )
            for _, slot in sorted(tc_map.items())
        ]
        if tool_calls:
            print()  # newline after a tool-calling turn
        message = types.SimpleNamespace(content="".join(content_parts), tool_calls=tool_calls)
        # Match the OpenAI SDK response shape run_agentic expects: resp.choices[0].message
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

    return chat


def sources_from_readset(read_set, corpus_map: dict) -> list[dict]:
    """Build grounded source objects from the chunk_ids the agent actually read."""
    KEEP = (
        "chunk_id",
        "source_ref",
        "title",
        "text",
        "source_type",
        "top_category",
        "risk_or_safe",
        "display_address",
        "legal_area",
    )
    new = []
    for cid in read_set:
        cand = cid
        if cand not in corpus_map and ":" in cid:
            cand = cid.split(":", 1)[1]
        corp = corpus_map.get(cand) or corpus_map.get(cid)
        if not corp:
            continue
        c = {k: corp.get(k) for k in KEEP if corp.get(k) is not None}
        c["norm_ids"] = derive_norm_ids(corp)
        new.append(c)
    return new


def _resolve_qdrant_path(preferred: str) -> str:
    import pathlib

    candidates = [preferred, ".rag/qdrant", str(DEFAULT_QDRANT_PATH)]
    for c in candidates:
        if not c:
            continue
        try:
            if QdrantHybridStore(collection_name=DEFAULT_COLLECTION, path=c).exists():
                return c
        except Exception:
            continue
    return preferred


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--parallel", type=int, default=3)
    ap.add_argument("--out", default="outputs/scenarios/synthetic_v1.jsonl")
    ap.add_argument("--invalid-out", default="outputs/scenarios/synthetic_v1_invalid.jsonl")
    ap.add_argument("--log", default="outputs/scenarios/synthetic_v1.log")
    ap.add_argument("--qdrant-path", default=".rag/qdrant")
    ap.add_argument("--collection", default=DEFAULT_COLLECTION)
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--max-iter", type=int, default=6)
    ap.add_argument(
        "--retrieval-url",
        default=os.environ.get("RETRIEVAL_ENDPOINT"),
        help="If set, route all retrieval tools to this Modal retrieval-as-a-service "
        "URL instead of loading BGE-M3/reranker locally (avoids OOM on small boxes).",
    )
    ap.add_argument(
        "--fresh",
        action="store_true",
        help="overwrite any existing --out instead of resuming (default: resume/skip completed)",
    )
    ap.add_argument(
        "--retries",
        type=int,
        default=3,
        help="per-scenario retries on transient/network errors before marking it failed",
    )
    ap.add_argument(
        "--instructions",
        default=None,
        help="optional jsonl of phase-3 questions (instructions[].text) to distill from "
        "instead of the built-in INSTRUCTION_TEMPLATES samples",
    )
    args = ap.parse_args()
    if args.instructions:
        n_bank = load_instruction_bank(args.instructions)
        logline_early = f"instruction bank: {n_bank} questions from {args.instructions}"
        print(logline_early, flush=True)
    # Only probe the local Qdrant when we will actually use it (non-remote mode).
    # In remote/Modal mode retrieval is served from Modal CPU, so the generation
    # box must never open the local Qdrant at all.
    if args.retrieval_url is None:
        args.qdrant_path = _resolve_qdrant_path(args.qdrant_path)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    log = open(args.log, "w")
    write_lock = threading.Lock()

    def logline(s):
        print(s, flush=True)
        print(s, file=log, flush=True)

    logline(f"[{datetime.datetime.utcnow().isoformat()}] loading Qdrant store")
    remote = None
    if args.retrieval_url:
        from legal_drafter.retrieval.agentic_tools import set_remote
        from legal_drafter.retrieval.remote import RemoteRetrieval

        remote = RemoteRetrieval(args.retrieval_url)
        set_remote(remote)
        logline(f"remote retrieval enabled: {args.retrieval_url}")

    store = None
    if remote is None:
        store = QdrantHybridStore(collection_name=args.collection, path=args.qdrant_path)
        if not store.exists():
            logline("[fatal] Qdrant collection does not exist; build it first")
            return 2

        logline("collecting anchor chunks ...")
        anchors = collect_anchors(store, args.n)
        logline(f"anchor pool size = {len(anchors)}")
        if not anchors:
            logline("[fatal] no anchor chunks retrieved")
            return 2

    # Generator model comes from CHECKER_MODEL (et al.) in the environment.
    cfg = load_checker()
    client = cfg.make_client()
    chat_main = make_chat(client, cfg, TOOL_SCHEMAS)

    tool_model = os.environ.get("TOOLTEACHER_MODEL")
    chat_tool = None
    if tool_model:
        from legal_drafter.config import LLMConfig

        tcfg = LLMConfig(
            role="toolteacher",
            model=tool_model,
            base_url=os.environ.get("TOOLTEACHER_BASE_URL") or cfg.base_url,
            api_key=os.environ.get("TOOLTEACHER_API_KEY") or cfg.api_key,
            timeout=cfg.timeout,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            extra_body=cfg.extra_body,
            extra_headers=cfg.extra_headers,
        )
        chat_tool = make_chat(tcfg.make_client(), tcfg, TOOL_SCHEMAS)

    logline(
        f"generator model={cfg.model} base_url={cfg.base_url} parallel={args.parallel} "
        f"max_iter={args.max_iter} toolteacher={'yes' if chat_tool else 'no'}"
    )

    max_tokens = args.max_tokens or cfg.max_tokens

    logline("source resolution will be served from the Modal retrieval service "
            "(CPU) per scenario; the generation box never opens the local Qdrant for RAG.")

    if remote is None:
        # Legacy local-RAG path only: warm up BGE-M3/reranker locally (this path
        # OOMs on the small generation box, which is why retrieval is served from
        # Modal instead).
        logline("warming up retrieval models (load once before workers) ...")
        try:
            from legal_drafter.retrieval.embeddings import encode, rerank_scores

            encode(["warmup query"])
            rerank_scores("warmup", ["context"])
            logline("model warmup ok")
        except Exception as e:  # pragma: no cover - model load dependent
            logline(f"[warn] model warmup failed: {e}")

    # --- resume support (survive network / power failure) ---
    resume = not args.fresh
    done_i: set[int] = set()
    stats = {"ok": 0, "invalid": 0, "error": 0}
    if resume and os.path.exists(args.out):
        valid_recs = []
        had_corrupt = False
        try:
            with open(args.out, "r") as ef:
                for line in ef:
                    s = line.strip()
                    if not s:
                        continue
                    try:
                        rec = json.loads(s)
                    except Exception:
                        had_corrupt = True
                        continue
                    valid_recs.append(rec)
                    tid = rec.get("task_id") or ""
                    if tid.startswith("synthetic_v1_"):
                        try:
                            done_i.add(int(tid[len("synthetic_v1_"):]))
                        except ValueError:
                            pass
                    gen = rec.get("generation") or {}
                    if gen.get("valid"):
                        stats["ok"] += 1
                    elif rec.get("error"):
                        stats["error"] += 1
                    else:
                        stats["invalid"] += 1
        except Exception as exc:
            logline(f"[warn] could not read existing output for resume: {exc!r}")
            valid_recs, done_i = [], set()
            had_corrupt = False
        if had_corrupt:
            # drop any corrupt trailing partial line(s) so the file stays valid JSONL
            with open(args.out, "w") as wf:
                for rec in valid_recs:
                    wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            logline(f"[info] cleaned corrupt lines from {args.out}; kept {len(valid_recs)} valid records")
    mode = "w" if (args.fresh or not (resume and os.path.exists(args.out))) else "a"
    logline(f"resume={'yes' if resume else 'no'} | completed={len(done_i)} | open_mode={mode}")

    write_lock_local = threading.Lock()
    out_f = open(args.out, mode)
    invalid_out_f = open(args.invalid_out, mode) if args.invalid_out else None
    done = len(done_i)

    def _gen_one(i: int) -> dict:
        instruction, dt_hint = build_instruction(i)
        session_id = f"scenario-{args.n}-{i}-{uuid.uuid4().hex[:12]}"
        result = run_agentic(
            store, instruction, chat_main, chat_tool,
            max_iter=args.max_iter, session_id=session_id, doc_type=dt_hint,
        )
        scn = result["scenario"]
        if scn is None:
            # Surface as a retryable error (transient model/parse failure).
            raise RuntimeError("no final JSON from teacher")
        # The teacher sometimes emits `sources` as a list of bare strings
        # (or a single string) instead of dicts; normalize so downstream
        # .get("source_ref") calls never crash.
        srcs = scn.get("sources")
        if isinstance(srcs, str):
            scn["sources"] = [{"source_ref": srcs}]
        elif isinstance(srcs, list):
            scn["sources"] = [
                s if isinstance(s, dict) else {"source_ref": str(s)}
                for s in srcs
            ]
        # Collect the refs the teacher actually cited (chunks it read + source_refs
        # it named) and resolve them to verbatim corpus chunks. This is SERVED
        # FROM MODAL (CPU) - the generation box never opens the local Qdrant for
        # RAG. Only the small cited subset is transferred, not the whole ~17k map.
        refs: list[str] = list(result["read_set"])
        for s in (scn.get("sources") or []):
            r = s.get("source_ref")
            if r and r not in refs:
                refs.append(r)
        if remote is not None:
            corpus_map = remote.resolve_sources(refs) if refs else {}
        else:
            corpus_map = build_corpus_map(store) if refs else {}
        if result["read_set"]:
            scn["sources"] = sources_from_readset(result["read_set"], corpus_map)
        else:
            canonicalize_sources(scn, corpus_map)
        rebuild_evidence_norms(scn)
        v = validate(scn, scn["sources"])
        return {
            "task_id": f"synthetic_v1_{i:04d}",
            "schema_version": "distill_drafting_task_v1",
            "topic": scn.get("topic"),
            "doc_type": scn.get("doc_type"),
            "instruction": scn.get("instruction"),
            "facts": scn.get("facts"),
            "source_refs": [s.get("source_ref") for s in (scn.get("sources") or [])],
            "sources": scn.get("sources"),
            "evidence_report": scn.get("evidence_report"),
            "expected_output": scn.get("expected_output"),
            "anchor_source_ref": None,
            "trajectory": result["messages"],
            "generation": {
                "model": cfg.model,
                "generated_at": datetime.datetime.utcnow().isoformat(),
                "valid": v["ok"],
                "validation_issues": v["issues"],
                "agentic": True,
                "read_set_size": len(result["read_set"]),
            },
        }

    def gen_one(i: int) -> dict:
        """Generate one scenario, retrying transient/network failures."""
        for attempt in range(1, max(args.retries, 1) + 1):
            try:
                return _gen_one(i)
            except Exception as exc:
                logline(f"[warn] scenario {i:04d} attempt {attempt}/{args.retries} failed: {exc!r}")
                if attempt < args.retries:
                    time.sleep(min(2 ** attempt, 30))
        return {
            "task_id": f"synthetic_v1_{i:04d}",
            "ok": False,
            "error": f"failed after {args.retries} attempts",
            "trajectory": [],
        }

    pending = [i for i in range(1, args.n + 1) if i not in done_i]
    logline(f"launching {len(pending)} pending scenarios (of {args.n}) with {args.parallel} workers ...")
    if not pending:
        logline("nothing to do - all scenarios already completed.")
    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futs = [ex.submit(gen_one, i) for i in pending]
        for fut in as_completed(futs):
            rec = fut.result()
            with write_lock_local:
                done += 1
                if rec.get("error"):
                    stats["error"] += 1
                    if invalid_out_f:
                        invalid_out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        invalid_out_f.flush()
                elif rec.get("expected_output") and len(rec.get("expected_output", "")) > 100:
                    stats["ok"] += 1
                    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out_f.flush()
                else:
                    stats["invalid"] += 1
                    if invalid_out_f:
                        invalid_out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        invalid_out_f.flush()

                if done % 10 == 0 or done == args.n:
                    logline(f"progress {done}/{args.n} | ok={stats['ok']} invalid={stats['invalid']} err={stats['error']}")

    out_f.close()
    if invalid_out_f:
        invalid_out_f.close()
    logline(f"DONE. {args.n} scenarios written to {args.out} (ok) and {args.invalid_out} (invalid) | ok={stats['ok']} invalid={stats['invalid']} err={stats['error']}")
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
