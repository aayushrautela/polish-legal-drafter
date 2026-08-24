"""PHASE BETA pilot: 3 parallel independent editor-runs over the same template.

Each run gets: primary umowa_zlecenia template + related templates (titles only)
+ a labeled clause bank pulled from the frozen v266 anchors, and must emit
STRUCTURED add/remove/modify ops, every op citing the bank item it grounded on.
No live tool loop in this pilot - the bank stands in for RAG (deterministic,
comparable across runs). Purpose: see whether independent runs converge or
diverge in what they add/remove (user's mixing/diversity question).

Usage (tmux): env PYTHONPATH=src .venv/bin/python run_beta.py
Inputs : beta_experiment/templates.jsonl           (from subagent)
         anchors.final_v266.jsonl                  (clause bank source)
Outputs: beta_experiment/run{1,2,3}.json           (raw structured edits)
         beta_experiment/compare.txt               (overlap/divergence digest)
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).parent
BANK_MAX = 12          # clauses shown per run
TEMPLATE_CHARS = 9000  # truncation guard


def load_env(path: str = ".env") -> dict:
    env = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()
    return env


def load_inputs():
    """Primary = longest umowa_zlecenia template; variants = 4 MORE VERSIONS OF
    THE SAME TYPE (user directive: variety from same-type variants only)."""
    primary, variants = "", []
    for l in open(HERE / "beta_experiment" / "templates.jsonl", encoding="utf-8"):
        r = json.loads(l)
        if r.get("role") == "primary":
            primary = r["text"][:TEMPLATE_CHARS]
        else:
            variants.append((f"variant#{r.get('variant_idx','?')}",
                             r["text"][:400]))
    bank, seen, pool = [], set(), []
    for l in open(HERE / "anchors.final_v266.jsonl", encoding="utf-8"):
        rec = json.loads(l)
        for c in rec.get("related_clauses", []):
            cid = c.get("source_ref") or c.get("chunk_id")
            t = " ".join((c.get("text") or "").split())
            if not cid or not t or t[:80] in seen:
                continue
            seen.add(t[:80])
            pool.append({"ref": cid, "text": t[:700]})
    step = max(1, len(pool) // BANK_MAX)
    bank = [pool[(i * step) % len(pool)] for i in range(min(BANK_MAX, len(pool)))]
    return primary, variants, bank


def build_prompt(tpl: str, related: list, bank: list) -> str:
    parts = ["[BASE TEMPLATE - umowa zlecenia]\n" + tpl]
    if related:
        parts.append("[SAME-TYPE TEMPLATE VARIANTS IN RAG (style references only)]\n"
                     + "\n".join(f"-- {d}: {t[:250]}" for d, t in related))
    parts.append("[CLAUSE BANK - you may ground edits ONLY on these]")
    for i, b in enumerate(bank, 1):
        parts.append(f"R{i} ref={b['ref']}\n{b['text']}")
    return "\n\n".join(parts) + (

        "\n\nTASK: You are tailoring this base template into an extended version "
        "for specific client situations. Using ONLY the clause bank above as "
        "grounding material, propose structured edits.\n"
        "Return STRICT JSON, no prose:\n"
        '{"edits":[{"op":"add|remove|modify","target":"<section or anchor phrase '
        'in base template>","new_text":"<full Polish clause text>","old_text":'
        '"<verbatim fragment being removed/modified, empty for add>","cite_ref":'
        '"<R# ref exactly as given>","reason":"<one line why this client case '
        'needs it>"}]}\n'
        "Rules: 3-6 edits. Every edit MUST cite one R#. new_text must be complete "
        "Polish contract language ready to paste into a contract. Do not invent "
        "material absent from the bank; adapt wording to the template's style."
    )


def run_one(idx: int, client, model: str, prompt: str, results: dict, errors: dict):
    try:
        # STREAMING is mandatory for this proxy+reasoning model - non-streaming
        # calls return empty content while reasoning tokens burn silently
        # (agent 1's make_chat pattern, scenario_gen.py:565-640).
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content":
                       "You are a senior Polish legal drafter editing a contract "
                       "template. Output ONLY strict JSON."},
                      {"role": "user", "content": prompt}],
            temperature=0.8,   # diversity across independent runs
            max_tokens=16000,
            timeout=600,
            stream=True,
        )
        parts = []
        for chunk in resp:
            if not chunk.choices:
                continue
            d = chunk.choices[0].delta
            if d is not None and getattr(d, "content", None):
                parts.append(d.content)
        txt = "".join(parts)
        m = re.search(r"\{.*\}", txt, re.S)
        parsed = json.loads(m.group(0)) if m else {"edits": [], "raw": txt[:2000]}
        results[f"run{idx}"] = parsed
    except Exception as e:
        errors[f"run{idx}"] = f"{type(e).__name__}: {e}"


def compare(results: dict, bank_refs: set) -> str:
    lines = []
    all_edits = {}
    for rid, data in sorted(results.items()):
        eds = data.get("edits", [])
        ops = {}
        for e in eds:
            ops[e.get("op")] = ops.get(e.get("op"), 0) + 1
        cited = sum(1 for e in eds if (e.get("cite_ref") or "") in bank_refs)
        lines.append(f"{rid}: {len(eds)} edits {ops} | cite_resolvable={cited}/{len(eds)}")
        all_edits[rid] = {(e.get("op"), " ".join((e.get('new_text') or '').split())[:120])
                          for e in eds}
    runs = list(all_edits)
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            shared = all_edits[runs[i]] & all_edits[runs[j]]
            lines.append(f"shared edits {runs[i]}∩{runs[j]}: {len(shared)}")
            for s in list(shared)[:3]:
                lines.append(f"   SHARED: {s[0]} | {s[1][:100]}")
    uniq = {}
    for rid, eds in all_edits.items():
        for e in eds:
            uniq.setdefault(e[1][:60], []).append(rid)
    solo = [k for k, v in uniq.items() if len(v) == 1]
    lines.append(f"unique-to-one-run edit keys: {len(solo)}/{len(uniq)}")
    return "\n".join(lines)


def main() -> int:
    env = load_env()
    from openai import OpenAI
    client = OpenAI(base_url=env["TEACHER_BASE_URL"], api_key=env["TEACHER_API_KEY"])
    model = env["TEACHER_MODEL"]

    tpl, related, bank = load_inputs()
    if not tpl:
        print("ERROR: templates.jsonl missing/empty - wait for subagent", file=sys.stderr)
        return 2
    print(f"template chars={len(tpl)} related_types={[d for d,_ in related]} bank={len(bank)}")

    prompt = build_prompt(tpl, related, bank)
    results, errors = {}, {}
    threads = []
    for i in (1, 2, 3):
        t = threading.Thread(target=run_one,
                             args=(i, client, model, prompt, results, errors))
        threads.append(t)

    started = threading.Event()

    def starter():
        started.set()
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    # fire the three threads as close together as possible
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for rid, data in results.items():
        (HERE / "beta_experiment" / f"{rid}.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    digest = compare(results, {b["ref"] for b in bank})
    (HERE / "beta_experiment" / "compare.txt").write_text(digest, encoding="utf-8")
    print(digest)
    if errors:
        print("ERRORS:", json.dumps(errors, indent=1), file=sys.stderr)
    return 0 if results and not errors else (1 if not results else 0)


if __name__ == "__main__":
    sys.exit(main())
