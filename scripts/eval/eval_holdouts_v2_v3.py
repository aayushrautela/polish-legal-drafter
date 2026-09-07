"""Evaluate holdout questions with base model and LoRA checkpoints via vLLM.

Two-phase pipeline (matches training distribution):
  Phase 1: Research loop with tools (stop at 5 successful tool calls)
  Phase 2: Drafting call (no tools) -> extract JSON contract

Uses RAG server (HTTP) for retrieval — NO local Qdrant.
Run via SSH tunnels: vLLM (:18000) + RAG (:10100).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

# Config — both via SSH tunnels to remote
VLLM_URL = "http://127.0.0.1:18000/v1/chat/completions"
RAG_URL = "http://127.0.0.1:10100"
DATA_PATH = "/home/aayush/Downloads/polish-legal-drafter/outputs/qa_pairs_all_final/sft_final_merged.jsonl"
HOLDOUT_IDS_PATH = "/home/aayush/Downloads/polish-legal-drafter/outputs/eval100/holdout_ids_true42.json"

MODELS = ["/workspace/models/Qwen3.5-2B", "ckpt25", "ckpt50", "ckpt100", "ckpt250"]

# Sampling defaults: Unsloth Qwen3.5 thinking-mode recommendations for general
# tasks (https://unsloth.ai/docs/models/qwen3.5/): top_p 0.95, top_k 20,
# min_p 0.0, presence_penalty 1.5, repetition_penalty 1.0 — except temperature,
# which is split per phase (cool research for tool discipline, hot draft for
# fluent prose). Both phases cap at 32768 tokens per Unsloth's adequate-output
# guidance and the eval rerun it documents. All values overridable via CLI
# args (see main()); the effective spec is recorded into every output row.
SAMPLING = {
    "research": {"temperature": 0.3, "max_tokens": 32768, "top_p": 0.95,
                 "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5,
                 "repetition_penalty": 1.0},
    "draft": {"temperature": 0.7, "max_tokens": 32768, "top_p": 0.95,
              "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5,
              "repetition_penalty": 1.0},
    "tool_choice": "auto",
}

DOC_TYPES = [
    "umowa_zlecenia", "umowa_o_dzielo", "umowa_najmu", "umowa_sprzedazy",
    "umowa_darowizny", "umowa_pozyczki", "umowa_dzierzawy",
    "umowa_ubezpieczenia", "pelnomocnictwo", "umowa_leasingu",
    "umowa_franczyzy", "umowa_kredytu", "umowa_dostawy",
    "umowa_przechowania", "wezwanie_do_zaplaty",
    "reklamacja_konsumenta", "odstapienie_konsumenta", "poreczenie",
    "kaucja_zaliczka", "umowa_faktoringu", "umowa_konsygnacji",
    "umowa_licencyjna", "umowa_o_prace", "umowa_o_zachowanie_poufnosci",
    "umowa_ramowa",
]

DOC_TYPE_DESCR_PL = {
    "umowa_zlecenia": "staranne działanie bez rezultatu, art.734-751 KC",
    "umowa_o_dzielo": "konkretny rezultat, art.627-646 KC",
    "umowa_najmu": "odpłatne używanie rzeczy, art.659-692 KC",
    "umowa_sprzedazy": "odpłatne przeniesienie towaru, art.535-602 KC",
    "umowa_darowizny": "bezpłatne przekazanie, forma notarialna, art.888-902 KC",
    "umowa_pozyczki": "przekazanie pieniędzy/rzeczy do używania, art.720-727 KC",
    "umowa_dzierzawy": "użytek rolny, art.693-709 KC",
    "umowa_ubezpieczenia": "ryzyko, polisa, art.805-834 KC",
    "pelnomocnictwo": "upoważnienie do działania w cudzym imieniu, art.95-109 KC",
    "umowa_leasingu": "dzierżawa z opcją zakupu rzeczy",
    "umowa_franczyzy": "know-how + znak towarowy, sieć dystrybucji",
    "umowa_kredytu": "przekazanie pieniędzy do zwrotu, art.69 Prawa bankowego",
    "umowa_dostawy": "cykliczna dostawa rzeczy ruchomych, art.605-612 KC",
    "umowa_przechowania": "przechowanie rzeczy, art.835-845 KC",
    "wezwanie_do_zaplaty": "żądanie zapłaty należności",
    "reklamacja_konsumenta": "reklamacja towaru/usługi, art.556-576 KC",
    "odstapienie_konsumenta": "odstąpienie od umowy zawartej na odległość, art.27 u.p.k.",
    "poreczenie": "gwarancja długu, art.876-887 KC",
    "kaucja_zaliczka": "zabezpieczenie wykonania umowy, zadatek",
    "umowa_faktoringu": "cesja wierzytelności z zaliczką",
    "umowa_konsygnacji": "towar do sprzedaży na rachunek komitenta, art.765-773 KC",
    "umowa_licencyjna": "prawo do używania znaku towarowego / IP",
    "umowa_o_prace": "stosunek pracy, art.KP",
    "umowa_o_zachowanie_poufnosci": "NDA, tajemnica przedsiębiorstwa",
    "umowa_ramowa": "umowa ogólna, brak konkretnych świadczeń",
}

DOC_TYPE_ENUM_DESC = (
    "Rodzaj dokumentu. MUSI być jedną z 25 wartości enum poniżej. "
    "Wybierz NAJBLIŻSZĄ intencji pytania użytkownika. "
    "Lista: " + "; ".join(
        f"{dt} ({DOC_TYPE_DESCR_PL[dt]})" for dt in DOC_TYPES
    )
)

# System prompts (from run_beta_doc.py / run_qa_pairs.py)
SYSTEM_RESEARCH = (
    "You are a Polish legal drafter. Use the retrieval tools to find "
    "relevant legal provisions, then write a COMPLETE contract in Polish.\n\n"
    "Rules:\n"
    "- Use only provisions found in the retrieved chunks. Do not invent.\n"
    "- After researching, return ONLY strict JSON: {\"summary\": \"<one-line "
    "scenario>\", \"contract\": \"<full contract>\"}"
)

SYSTEM_DRAFT = (
    "You are a Polish legal drafter. Use the retrieval tools to find "
    "relevant legal provisions, then write a COMPLETE contract in Polish.\n\n"
    "Rules:\n"
    "- Use only provisions found in the retrieved chunks. Do not invent.\n"
    "- After researching, return ONLY strict JSON: {\"summary\": \"<one-line "
    "scenario>\", \"contract\": \"<full contract>\"}"
)

FORCED_DRAFT = (
    "Write the COMPLETE contract in Polish. "
    "Return ONLY strict JSON: "
    '{\"summary\": \"<one-line>\", \"contract\": \"<COMPLETE contract>\"}'
)


def execute_tool(name: str, args: dict) -> list:
    """Execute a retrieval tool call via RAG HTTP server. Returns list of result dicts."""
    if name == "get_template":
        doc_type = args.get("doc_type", "")
        resp = httpx.post(f"{RAG_URL}/get_template", json={"doc_type": doc_type}, timeout=30)
        resp.raise_for_status()
        return resp.json()

    elif name == "keyword_search":
        query = args.get("query", "")
        limit = args.get("limit", 10)
        resp = httpx.post(f"{RAG_URL}/keyword_search", json={"query": query, "limit": limit}, timeout=30)
        resp.raise_for_status()
        return resp.json()

    elif name == "semantic_search":
        query = args.get("query", "")
        limit = args.get("limit", 10)
        resp = httpx.post(f"{RAG_URL}/semantic_search", json={"query": query, "limit": limit}, timeout=30)
        resp.raise_for_status()
        return resp.json()

    elif name == "chunk_read":
        chunk_ids = args.get("chunk_ids", [])
        if not chunk_ids:
            return [{"error": "no chunk_ids"}]
        resp = httpx.post(f"{RAG_URL}/chunk_read", json={"chunk_ids": chunk_ids}, timeout=30)
        resp.raise_for_status()
        return resp.json()

    else:
        return [{"error": f"unknown tool {name}"}]


def is_successful_result(result: list) -> bool:
    """A tool call counts as successful only if it returned real content."""
    if not isinstance(result, list) or not result:
        return False
    for r in result:
        if isinstance(r, dict) and "error" not in r and (
            r.get("text") or r.get("chunk_id")
            or r.get("kind") == "structural_template"):
            return True
    return False


def call_vllm(messages: list, tools: list | None, model: str, max_tokens: int | None = None, temp: float | None = None) -> dict:
    """Call vLLM chat completions API."""
    # Phase split: research (tools) low temp for doc_type, draft (no tools) high temp.
    # All sampling comes from SAMPLING (CLI-overridable); explicit args win.
    is_research = tools is not None
    phase = "research" if is_research else "draft"
    cfg = SAMPLING[phase]
    temperature = temp if temp is not None else cfg["temperature"]
    max_tokens = max_tokens if max_tokens is not None else cfg["max_tokens"]
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": cfg["top_p"],
        "presence_penalty": cfg["presence_penalty"],
        "extra_body": {"top_k": cfg["top_k"], "min_p": cfg["min_p"],
                       "repetition_penalty": cfg["repetition_penalty"]},
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    resp = httpx.post(VLLM_URL, json=payload, timeout=600)
    resp.raise_for_status()
    result = resp.json()["choices"][0]["message"]
    # Qwen3.5 thinking model: response may have "reasoning" (not "reasoning_content")
    reasoning = result.get("reasoning", "") or result.get("reasoning_content", "")
    content = result.get("content", "") or ""
    return {
        "content": content,
        "reasoning": reasoning,
        "tool_calls": result.get("tool_calls", []),
    }


def extract_json(text: str):
    """Extract JSON from text (handles raw newlines in string values)."""
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```", 2)[1]
        if t.startswith("json"):
            t = t[4:]
    m = re.search(r"\{.*\}", t, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        pass
    # Retry after escaping raw control chars inside string values
    try:
        escaped = _escape_json_string_values(m.group(0))
        return json.loads(escaped)
    except Exception:
        return None


def _escape_json_string_values(s):
    """Escape raw control chars inside JSON string values."""
    out = []
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        if c == '"':
            j = i + 1
            while j < n:
                c2 = s[j]
                if c2 == '\\':
                    j += 2
                    continue
                if c2 == '"':
                    break
                j += 1
            val = s[i:j + 1]
            val = val.replace('\r', '\\r').replace('\n', '\\n').replace('\t', '\\t')
            out.append(val)
            i = j + 1
        else:
            out.append(c)
            i += 1
    return ''.join(out)


def build_transcript(messages: list) -> str:
    """Build a text transcript of the research history (for Phase 2 drafting)."""
    transcript = []
    for m in messages:
        role = m["role"]
        if role == "system":
            continue
        if role == "user":
            transcript.append(f"USER TASK: {m.get('content','')}")
        elif role == "assistant":
            if m.get("content"):
                transcript.append(f"DRAFTER: {m['content']}")
            for tc in (m.get("tool_calls") or []):
                fn = tc["function"]["name"]
                args = tc["function"]["arguments"]
                transcript.append(f"DRAFTER CALLED {fn}({args})")
        elif role == "tool":
            try:
                parsed = json.loads(m.get("content") or "[]")
            except Exception:
                parsed = None
            if isinstance(parsed, list):
                items = []
                for r in parsed:
                    if isinstance(r, dict):
                        r = dict(r)
                        if "text" in r and len(r["text"]) > 3000 and r.get("kind") != "structural_template":
                            r["text"] = r["text"][:3000] + " \u2026[truncated]"
                        items.append(r)
                    else:
                        items.append(r)
                blob = json.dumps(items, ensure_ascii=False)
            else:
                blob = (m.get("content") or "")[:6000]
            transcript.append(f"RESULT: {blob}")
    return "\n".join(transcript)


def run_holdout(row: dict, model: str, tools: list) -> dict:
    """Run two-phase pipeline for one holdout with one model."""
    # Only send system + user — let the model do the full research loop
    raw_messages = row["messages"]
    messages = []
    for m in raw_messages[:2]:  # [system, user] only
        role = m.get("role")
        if role in ("system", "user"):
            messages.append({"role": role, "content": m.get("content", "")})

    row_id = row["id"]

    # --- Phase 1: Research (tool-calling loop) ---
    success = 0
    total_calls = 0
    max_iter = 10
    for it in range(max_iter):
        try:
            response = call_vllm(messages, tools, model)
        except Exception as e:
            return {"id": row_id, "model": model, "sampling": SAMPLING,
                    "error": f"P1 turn {it}: {str(e)}"}

        # Append assistant message
        assistant_msg = {
            "role": "assistant",
            "content": response.get("content", ""),
        }
        if response.get("reasoning"):
            assistant_msg["reasoning_content"] = response["reasoning"]
        if response.get("tool_calls"):
            assistant_msg["tool_calls"] = response["tool_calls"]
        messages.append(assistant_msg)

        # If no tool calls, stop research
        if not response.get("tool_calls"):
            break

        # Execute tool calls
        for tc in response["tool_calls"]:
            fn = tc["function"]
            try:
                args = json.loads(fn["arguments"])
            except json.JSONDecodeError:
                args = {}
            tool_result = execute_tool(fn["name"], args)
            total_calls += 1
            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "name": fn["name"],
                "content": json.dumps(tool_result, ensure_ascii=False),
            })
            if is_successful_result(tool_result):
                success += 1

        if success >= 5:
            break

    # --- Phase 2: Drafting (no tools) -> contract ---
    transcript_text = build_transcript(messages)

    draft_messages = [
        {"role": "system", "content": SYSTEM_DRAFT},
        {"role": "user", "content": (
            "RESEARCH HISTORY (use ONLY this material):\n" + transcript_text +
            "\n\nWrite the complete contract now."
        )},
    ]

    try:
        dresp = call_vllm(draft_messages, None, model)
    except Exception as e:
        # Truncate transcript if too long
        error_str = str(e)
        if "400" in error_str and len(transcript_text) > 10000:
            transcript_text = transcript_text[:10000] + "\n...[truncated]"
            draft_messages[1]["content"] = (
                "RESEARCH HISTORY (use ONLY this material):\n" + transcript_text +
                "\n\nWrite the complete contract now."
            )
            try:
                dresp = call_vllm(draft_messages, None, model)
            except Exception as e2:
                return {"id": row_id, "model": model, "sampling": SAMPLING,
                        "error": f"P2 draft retry: {str(e2)}"}
        else:
            return {"id": row_id, "model": model, "sampling": SAMPLING,
                    "error": f"P2 draft: {error_str}"}

    contract = dresp.get("content", "") or dresp.get("reasoning", "")
    summary = ""
    draft_reasoning = dresp.get("reasoning", "")

    return {
        "id": row_id,
        "model": model,
        "sampling": SAMPLING,
        "summary": summary,
        "contract": contract,
        "reasoning_content": draft_reasoning,
        "raw_research_messages": messages,
        "raw_draft_response": dresp,
        "n_tool_calls": total_calls,
        "n_successful": success,
        "empty": not contract,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=None, help="Number of questions to evaluate (default: all if --input, else 10)")
    parser.add_argument("--parallel", type=int, default=3, help="Parallel workers")
    parser.add_argument("--models", nargs="+", default=None, help="Model paths to evaluate")
    parser.add_argument("--input", default=None, help="Path to a jsonl of eval rows (id + messages). Runs all rows in it; overrides the built-in holdout selection.")
    parser.add_argument("--out", default="outputs/eval_results_v2.jsonl")
    # Sampling (defaults: Unsloth Qwen3.5 thinking-mode recommendations,
    # https://unsloth.ai/docs/models/qwen3.5/ , except the 2-stage temps).
    # The effective spec is recorded into every output row.
    parser.add_argument("--research-temp", type=float, default=0.3)
    parser.add_argument("--draft-temp", type=float, default=0.7)
    parser.add_argument("--research-max-tokens", type=int, default=32768)
    parser.add_argument("--draft-max-tokens", type=int, default=32768)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--presence", type=float, default=1.5)
    parser.add_argument("--repetition", type=float, default=1.0)
    args = parser.parse_args()

    for phase, temp, cap in (("research", args.research_temp, args.research_max_tokens),
                             ("draft", args.draft_temp, args.draft_max_tokens)):
        SAMPLING[phase].update(temperature=temp, max_tokens=cap,
                               top_p=args.top_p, top_k=args.top_k,
                               min_p=args.min_p,
                               presence_penalty=args.presence,
                               repetition_penalty=args.repetition)

    if args.models:
        MODELS[:] = args.models

    # Verify services
    print("Checking vLLM...", end=" ")
    r = httpx.get(VLLM_URL.replace("/chat/completions", "/models"), timeout=5)
    print("OK" if r.status_code == 200 else f"FAIL ({r.status_code})")
    print("Checking RAG...", end=" ")
    r = httpx.get(f"{RAG_URL}/health", timeout=5)
    print("OK" if r.status_code == 200 else f"FAIL ({r.status_code})")

    # Load eval rows: either from --input (any jsonl of {id, messages} rows) or
    # the built-in 42-holdout selection from the SFT pool. The --input branch
    # is what we use to run the 58 freshly-authored ext_* questions on top of
    # the existing clean 42-eval.
    if args.input:
        with open(args.input) as f:
            holdouts = [json.loads(l) for l in f]
        seen = set()
        deduped = []
        for r in holdouts:
            if r["id"] in seen:
                continue
            seen.add(r["id"])
            deduped.append(r)
        holdouts = deduped
        print(f"Loaded {len(holdouts)} eval rows from --input {args.input}")
        if args.n is not None and args.n < len(holdouts):
            holdouts = holdouts[: args.n]
    else:
        # Load data
        with open(DATA_PATH) as f:
            lines = [json.loads(l) for l in f]
        with open(HOLDOUT_IDS_PATH) as f:
            holdout_ids = json.load(f)
        if isinstance(holdout_ids, dict):
            holdout_ids = holdout_ids["ids"]

        # Select by ID VALUE (row "id" field), never by line index.
        # Index-based selection was the contamination bug (see
        # outputs/legacy/contaminated_holdout_evals_20260903/README.md).
        rows_by_id = {}
        for row in lines:
            rows_by_id.setdefault(row["id"], row)
        missing = [i for i in holdout_ids if i not in rows_by_id]
        if missing:
            raise SystemExit(f"Holdout ids not found in data: {missing}")
        holdouts = [rows_by_id[i] for i in holdout_ids]

        # Subset - for small n, test failed IDs (406,194,1534,1707,227) positions [20,6,12,16,22]
        if args.n is not None and args.n <= 5:
            failed_pos = [20, 6, 12, 16, 22][:args.n]
            holdouts = [holdouts[i] for i in failed_pos]
        elif args.n is not None and args.n < len(holdouts):
            holdouts = holdouts[: args.n]

    # Tool schemas: from the SFT pool's first row (the only place the canonical
    # tool schema lives). The doc_type enum patch below still runs so any
    # new doc_types referenced in --input rows stay in the enum.
    with open(DATA_PATH) as f:
        first_row = json.loads(f.readline())
    tools = first_row.get("tools", [])

    for tool in tools:
        if tool.get("function", {}).get("name") == "get_template":
            params = tool["function"].setdefault("parameters", {})
            props = params.setdefault("properties", {})
            if "doc_type" in props:
                props["doc_type"]["type"] = "string"
                props["doc_type"]["enum"] = DOC_TYPES
                props["doc_type"]["description"] = DOC_TYPE_ENUM_DESC
            break

    total = len(holdouts) * len(MODELS)
    done = 0

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    start = time.time()

    with ThreadPoolExecutor(max_workers=args.parallel) as pool, open(args.out, "w") as out_f:
        futures = {}
        for row in holdouts:
            for model in MODELS:
                f = pool.submit(run_holdout, row, model, tools)
                futures[f] = (row["id"], model)

        for f in as_completed(futures):
            row_id, model = futures[f]
            done += 1
            try:
                result = f.result()
            except Exception as e:
                result = {"id": row_id, "model": model, "error": traceback.format_exc()}

            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_f.flush()

            status = "ERROR" if "error" in result else ("EMPTY" if result.get("empty") else "OK")
            print(f"[{done}/{total}] id={row_id} model={model} {status} "
                  f"calls={result.get('n_tool_calls', '?')} "
                  f"success={result.get('n_successful', '?')} "
                  f"contract={len(result.get('contract', ''))}")

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s. {done} results written to {args.out}")

    # Summary
    print(f"\nAll done. Results written to {args.out}")


if __name__ == "__main__":
    main()
