"""Phase 3 - synthesize layperson ("avg Joe") instructions from the mutated set.

For EVERY mutated anchor (output of phase 2) generate --per-anchor natural,
first-person Polish instructions a layperson would actually type. The
instruction states the SITUATION and NEEDS (never clause text, never
citations, no legalese); the underlying conditions from the anchor + adds +
removes must be recoverable from what the user asks for.

Mechanism mirrors phase2_batch.py: doc_type grouping (template paid once),
batch anchors per call, ONE FORCED tool call submit_instructions carrying all
anchors' variants in a single args blob (no parallel-call stitching), streaming
accumulation, validation, resume + single-anchor retry pass.

Usage:
  PYTHONPATH=src RETRIEVAL_ENDPOINT=... .venv/bin/python -m legal_drafter.phase3_instruct \
      --in  outputs/anchors/anchor_pool.phase2.jsonl \
      --out outputs/anchors/anchor_pool.phase3.jsonl \
      --batch 5 --per-anchor 2 --parallel 4
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from legal_drafter.config import load_checker  # noqa: E402

TRUNC_TEMPLATE = 2200
TRUNC_ANCHOR = 1300

TOOL_NAME = "submit_instructions"

# A layperson does not cite statutes or paragraphs.
CITE_RE = re.compile(r"(art\.\s*\d|Dz\.U\.?|§|ustaw[ay]\s*z\s*dnia)", re.IGNORECASE)
MIN_LEN = 40
MAX_LEN = 700

SYSTEM_INSTRUCT = (
    "Jesteś ekspertem od polskich umów, który zamienia wymagania prawne w "
    "naturalne prośby zwykłych ludzi. Dostajesz WSPÓLNY SZABLON umowy (jeśli "
    "jest) i kilka ANCHOR situations oznaczonych A1..An. Dla każdej kotwicy "
    "znasz jej klauzulę oraz warunki DODAJ i USUŃ wypracowane w poprzednim "
    "etapie.\n"
    "Dla KAŻDEJ kotwicy A1..An napisz dokładnie {n} instrukcji (variants): "
    "to co zwykły człowiek bez wiedzy prawniczej wpisałby w formularz, żeby "
    "dostać dokument spełniający te warunki.\n"
    "ZASADY (łamanie psuje dane):\n"
    "1. Głos pierwszoosobowy, codzienny język. Zero cytátów z ustaw: żadnych "
    "'art.', '§', 'Dz.U.', 'ustawy z dnia'. Zero żargonu notarialnego jeśli "
    "da się go powiedzieć prosto.\n"
    "2. NIE formułuj obowiązków ani klauzul wprost ('wynajmujący ma obowiązek "
    "utrzymywać lokal...', 'umowa musi przewidywać...'). Człowiek nie zna "
    "przepisów - mówi o SWOJEJ SYTUACJI i POTRZEBACH: 'wynajmuję lokal pod "
    "sklep, boję się, że zostawię dach do remontu, chcę mieć jasność, kto za "
    "co płaci'. Zamieniaj każdy warunek na intencję, obawę lub prośbę.\n"
    "3. WSZYSTKIE kluczowe warunki kotwicy (jej klauzula + dodane warunki + "
    "usunięte fragmenty szablonu) muszą BYĆ WYCZUWALNE w potrzebach: kto jest "
    "stroną, co przedmiot, jakie zabezpieczenia, terminy, kary, forma - ale "
    "jako to, czego człowiek chce lub się obawia, nie jako punkty umowy.\n"
    "4. Style: pierwszy wariant 'prosta' - 1-3 krótkie zdania, czysta "
    "sytuacja + główna potrzeba, potocznie, bez wyliczania szczegółów. Drugi "
    "'szczegółowa' - ZAWSZE nadal głos człowieka (zaczyna się od sytuacji "
    "lub potrzeby, nigdy od treści przepisu/obowiązku); może doprecyzować "
    "kilka rzeczy (kwota, termin, zakres, kto co pokrywa) w formie próśb lub "
    "oczekiwań ('chcę mieć jasne, że...', 'zależy mi, żeby naprawy dachu i "
    "instalacji zostawały po stronie właściciela'). NIGDY nie stwierdzaj "
    "treści klauzul ('wynajmujący ma obowiązek...', 'umowa przewiduje...', "
    "'najemca ponosi koszty...') - to głos prawnika, nie użytkownika. "
    "Zwięźle, 2-5 zdań, bez rozwlekłości.\n"
    "5. Nie wymyślaj faktów, których nie ma w kotwicy ani warunkach.\n"
    "Odpowiadasz WYŁĄCZNIE jednym wywołaniem submit_instructions."
)

TOOLS_P3 = [
    {
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "description": (
                "Submit ALL layperson instructions for EVERY anchor (A1..An) "
                "in one call. Exactly one call per response."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "instructions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "anchor_id": {"type": "string",
                                              "description": "anchor id, e.g. A1"},
                                "variants": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "style": {"type": "string",
                                                      "description":
                                                      "np. 'prosta' / 'szczegółowa'"},
                                            "text": {"type": "string",
                                                     "description":
                                                     "instrukcja użytkownika po polsku"},
                                        },
                                        "required": ["style", "text"],
                                    },
                                },
                            },
                            "required": ["anchor_id", "variants"],
                        },
                    },
                },
                "required": ["instructions"],
            },
        },
    },
]


def _trunc(text: str, n: int) -> str:
    if text is None:
        return ""
    text = text.strip()
    return text if len(text) <= n else text[:n - 1].rstrip() + "…"


def _chat_batch_tool(client, cfg, messages, max_tokens: int,
                     temperature: float = 0.3):
    """Force ONE submit_instructions call; returns parsed arguments dict."""
    base = dict(model=cfg.model, messages=messages, tools=TOOLS_P3,
                temperature=temperature, max_tokens=max_tokens, stream=True)
    if cfg.extra_body:
        base["extra_body"] = cfg.extra_body
    if cfg.extra_headers:
        base["extra_headers"] = cfg.extra_headers

    def accumulate(kwargs):
        args_buf = ""
        name = None
        for chunk in client.chat.completions.create(**kwargs):
            if not chunk.choices:
                continue
            tcs = chunk.choices[0].delta.tool_calls
            if not tcs:
                continue
            for t in tcs:
                if t.function and t.function.name:
                    name = t.function.name
                if t.function and t.function.arguments:
                    args_buf += t.function.arguments
        return name, args_buf

    attempts = [
        dict(base, tool_choice={"type": "function", "function": {"name": TOOL_NAME}}),
        dict(base, tool_choice="auto",
             messages=messages + [{"role": "user",
                                   "content": f"Now call {TOOL_NAME} once with "
                                              "every anchor's variants."}]),
    ]
    last_err: Exception | None = None
    for kwargs in attempts:
        try:
            name, args_buf = accumulate(kwargs)
            if not args_buf.strip():
                last_err = RuntimeError("empty tool arguments")
                continue
            try:
                parsed = json.loads(args_buf)
            except Exception:
                m = re.search(r"\{.*\}", args_buf, re.DOTALL)
                parsed = json.loads(m.group(0)) if m else {}
                if not isinstance(parsed, dict) or not parsed:
                    raise ValueError(
                        f"unparseable arguments ({len(args_buf)} chars)")
            if name and name != TOOL_NAME and isinstance(parsed, list):
                parsed = {"instructions": parsed}
            if not isinstance(parsed, dict):
                raise ValueError("arguments are not an object")
            return parsed
        except TypeError as e:
            last_err = e
            continue
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
    raise last_err if last_err else RuntimeError("no tool call produced")


def build_prompt(batch: list[dict], template_text: str | None,
                 per_anchor: int) -> str:
    parts = []
    if template_text:
        parts.append("[SZABLON]\n" + _trunc(template_text, TRUNC_TEMPLATE))
    else:
        parts.append("[SZABLON]\n(brak wspólnego szablonu dla tej grupy)")
    for i, rec in enumerate(batch, 1):
        adds = rec.get("adds") or []
        removes = rec.get("removes") or []
        block = (f"[KOTWICA A{i}] chunk_id={rec['chunk_id']}\n"
                 f"Klauzula kotwicy:\n{_trunc(rec.get('text') or '', TRUNC_ANCHOR)}")
        if adds:
            cl = "\n".join(f"- {a.get('condition', '')} "
                           f"(źródło: {a.get('cite_ref', '?')})" for a in adds[:4])
            block += f"\nWarunki do DODANIA (muszą wynikać z instrukcji):\n{cl}"
        if removes:
            cl = "\n".join(f"- {r.get('template_clause', '')}" for r in removes[:4])
            block += f"\nFragmenty szablonu do USUNIĘCIA (instrukcja nie powinna "
            block += f"zakładać ich istnienia):\n{cl}"
        parts.append(block)
    return ("\n\n".join(parts)
            + f"\n\nWywołaj submit_instructions RAZ: tablica instructions dla "
              f"KAŻDEJ kotwicy A1..A{len(batch)}, każdy z dokładnie {per_anchor} "
              "wariantami (pierwszy 'prosta').")


def _validate_variants(variants: list, per_anchor: int) -> list:
    out = []
    seen_texts = set()
    for v in variants or []:
        text = (v.get("text") or "").strip()
        style = (v.get("style") or "").strip() or "wariant"
        if not (MIN_LEN <= len(text) <= MAX_LEN):
            continue
        if CITE_RE.search(text):
            continue  # laypeople don't cite statutes
        key = text[:80].lower()
        if key in seen_texts:
            continue
        seen_texts.add(key)
        out.append({"style": style, "text": text})
        if len(out) >= per_anchor:
            break
    return out


def process_batch(client, cfg, batch: list[dict], template_text: str | None,
                  per_anchor: int, max_tokens: int, retries: int) -> dict:
    prompt = build_prompt(batch, template_text, per_anchor)
    sys_prompt = SYSTEM_INSTRUCT.replace("{n}", str(per_anchor))
    messages = [{"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt}]
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            parsed = _chat_batch_tool(client, cfg, messages, max_tokens)
            by_anchor = {}
            items = parsed.get("instructions")
            if isinstance(items, dict):
                items = [items]
            for m in items or []:
                if not isinstance(m, dict):
                    continue
                aid = str(m.get("anchor_id") or "").strip()
                mo = re.match(r"A?(\d+)", aid)
                if not mo:
                    continue
                by_anchor.setdefault(int(mo.group(1)), []).append(m)
            res = {}
            for i, rec in enumerate(batch, 1):
                entries = by_anchor.get(i, [])
                variants = []
                for m in entries:
                    variants += _validate_variants(m.get("variants"), per_anchor)
                    if len(variants) >= per_anchor:
                        break
                if variants:
                    res[rec["chunk_id"]] = {"instructions": variants}
                else:
                    res[rec["chunk_id"]] = {"instructions": [],
                                            "phase3_error":
                                            "no valid instructions"}
            return res
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(min(2 * attempt, 10))
    return {rec["chunk_id"]: {"instructions": [],
                              "phase3_error": last_err or "unknown"}
            for rec in batch}


def _rewrite_line(out: Path, chunk_id: str, new_rec: dict) -> None:
    lines = out.read_text(encoding="utf-8").splitlines()
    out_lines = []
    replaced = False
    for l in lines:
        if not l.strip():
            continue
        try:
            r = json.loads(l)
        except Exception:
            out_lines.append(l)
            continue
        if r.get("chunk_id") == chunk_id and not replaced:
            out_lines.append(json.dumps(new_rec, ensure_ascii=False))
            replaced = True
        else:
            out_lines.append(l)
    if not replaced:
        out_lines.append(json.dumps(new_rec, ensure_ascii=False))
    out.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp",
                    default="outputs/anchors/anchor_pool.phase2.jsonl")
    ap.add_argument("--out", default="outputs/anchors/anchor_pool.phase3.jsonl")
    ap.add_argument("--batch", type=int, default=5)
    ap.add_argument("--per-anchor", type=int, default=2)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--retry-pass", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=12000)
    args = ap.parse_args()

    cfg = load_checker()
    client = cfg.make_client()
    inp = Path(args.inp)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    raw = [json.loads(l) for l in inp.read_text(encoding="utf-8").splitlines()
           if l.strip()]
    # last write wins (retry pass may have rewritten earlier error lines)
    recs = {}
    for r in raw:
        recs[r["chunk_id"]] = r
    # skip records whose phase-2 mutation errored (nothing to instruct FROM)
    usable = [r for r in recs.values()
              if not r.get("mutation_error") and (r.get("adds") or r.get("removes"))]

    done = set()
    if out.exists():
        for l in out.read_text(encoding="utf-8").splitlines():
            if l.strip():
                try:
                    r = json.loads(l)
                    if not r.get("phase3_error") and r.get("instructions"):
                        done.add(r["chunk_id"])
                except Exception:
                    pass
    pending = [r for r in usable if r["chunk_id"] not in done]
    print(f"records={len(recs)} usable={len(usable)} done={len(done)} "
          f"pending={len(pending)}", flush=True)

    groups = defaultdict(list)
    for r in pending:
        groups[r.get("doc_type_primary") or "other"].append(r)

    batches = []
    for dt, items in groups.items():
        template_text = items[0].get("template_text") if dt != "other" else None
        for k in range(0, len(items), args.batch):
            batches.append((dt, template_text, items[k:k + args.batch]))
    print(f"doc_types={len(groups)} batches={len(batches)}", flush=True)

    write_lock = Lock()
    errs = 0
    written = 0
    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futs = {ex.submit(process_batch, client, cfg, b[2], b[1],
                          args.per_anchor, args.max_tokens,
                          args.retries): b for b in batches}
        completed = 0
        for fut in as_completed(futs):
            dt, template_text, batch = futs[fut]
            try:
                res = fut.result()
            except Exception as e:  # noqa: BLE001
                res = {r["chunk_id"]: {"instructions": [],
                                       "phase3_error": str(e)} for r in batch}
            with write_lock:
                with out.open("a", encoding="utf-8") as f:
                    for rec in batch:
                        cid = rec["chunk_id"]
                        got = res.get(cid, {"instructions": [],
                                            "phase3_error": "missing"})
                        out_rec = dict(rec)
                        out_rec["instructions"] = got.get("instructions", [])
                        if got.get("phase3_error"):
                            out_rec["phase3_error"] = got["phase3_error"]
                            errs += 1
                        out_rec["phase3_dt"] = dt
                        out_rec["phase3_source"] = "batch_tool"
                        f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                        written += 1
            completed += 1
            if completed % 5 == 0 or completed == len(batches):
                print(f"progress batches {completed}/{len(batches)} "
                      f"written={written} errors={errs}", flush=True)

    # Retry pass for anchors with no valid instructions.
    errored = []
    if out.exists():
        for l in out.read_text(encoding="utf-8").splitlines():
            if not l.strip():
                continue
            try:
                r = json.loads(l)
            except Exception:
                continue
            if r.get("phase3_error"):
                errored.append(r["chunk_id"])
    print(f"retry pass: {len(errored)} errored anchors", flush=True)
    rdone = 0
    by_id = recs
    for cid in errored:
        rec = by_id.get(cid)
        if not rec:
            continue
        tmpl = rec.get("template_text") if (
            rec.get("doc_type_primary")
            and rec.get("doc_type_primary") != "other") else None
        mut = None
        for _ in range(max(1, args.retry_pass)):
            res = process_batch(client, cfg, [rec], tmpl, args.per_anchor,
                                args.max_tokens, 1)
            mut = res.get(cid)
            if mut and not mut.get("phase3_error"):
                break
        if mut is None:
            mut = {"instructions": [], "phase3_error": "retry failed"}
        out_rec = dict(rec)
        out_rec["instructions"] = mut.get("instructions", [])
        if mut.get("phase3_error"):
            out_rec["phase3_error"] = mut["phase3_error"]
        else:
            out_rec.pop("phase3_error", None)
        out_rec["phase3_dt"] = rec.get("doc_type_primary") or "other"
        out_rec["phase3_source"] = "batch_tool_retry"
        _rewrite_line(out, cid, out_rec)
        rdone += 1
        if rdone % 10 == 0 or rdone == len(errored):
            print(f"retry progress {rdone}/{len(errored)}", flush=True)

    final_err = 0
    total_instr = 0
    if out.exists():
        seen = {}
        for l in out.read_text(encoding="utf-8").splitlines():
            if not l.strip():
                continue
            try:
                r = json.loads(l)
            except Exception:
                continue
            seen[r["chunk_id"]] = r
        final_err = sum(1 for r in seen.values() if r.get("phase3_error"))
        total_instr = sum(len(r.get("instructions") or [])
                          for r in seen.values())
    print(f"DONE. written={written} first_pass_errors={errs} "
          f"final_errors={final_err} total_instructions={total_instr}",
          flush=True)


if __name__ == "__main__":
    main()
