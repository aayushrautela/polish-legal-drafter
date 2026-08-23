"""Agentic tool-calling loop for scenario generation (and later inference).

Drives the teacher model with native OpenAI-style tool calling over the A-RAG
hierarchical retrieval tools (keyword_search / semantic_search / chunk_read),
recording the full native transcript so it can be distilled into the LoRA
(perfect train/inference distribution match, KAFT).

Loop shape (seq 32): PLAN -> parallel batched retrieve -> chunk_read -> WRITE
-> optional verify, bounded by an iteration cap and a context tracker. Mixed
teacher (seq 29) is supported: ``chat_tool`` (a cheaper model) is used for the
turns that interpret tool observations and emit the next queries; ``chat_main``
(the strong teacher) handles PLAN and WRITE.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Optional

from legal_drafter.retrieval.agentic_tools import TOOL_SCHEMAS, execute_tool


def _extract_json(text: str):
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


def _tc_to_dict(msg) -> list[dict]:
    out = []
    for tc in msg.tool_calls or []:
        out.append(
            {
                "id": tc.id,
                "type": getattr(tc, "type", "function"),
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
        )
    return out


def _build_system_prompt(schema: str, doc_type_hint: str) -> str:
    return (
        "You are a senior Polish legal drafter building a high-quality distillation "
        "dataset for a legal-language model. You have four retrieval tools and MUST "
        "ground every scenario in REAL Polish legal sources you retrieve yourself:\n"
        "- keyword_search: exact lexical match for known terms (article numbers like "
        "'art. 27', legal terms, entity names, statute refs like 'KC:483').\n"
        "- semantic_search: conceptual/dense retrieval for paraphrased queries.\n"
        "- chunk_read: fetch the FULL text of specific chunks by chunk_id (always read "
        "full text before citing).\n"
        "- get_template: fetch the REAL, sourced (CC-BY-4.0) Polish template/guide for the "
        "doc_type - its section structure, standard clause language, and real statutory "
        "references. Use it as the authoritative structural AND substantive skeleton.\n\n"
        "TEMPLATE DISCIPLINE (critical):\n"
        "- ALWAYS call get_template for the doc_type as your FIRST retrieval step, to obtain "
        "the authoritative skeleton (section order, real clauses, real statutory refs).\n"
        "- Follow the template's SECTION ORDER and reuse its correct clause language and "
        "statutory references (e.g. art. 659-692 KC). These are real, sourced rules - cite "
        "them as authority alongside the retrieved sources.\n"
        "- ADAPT every clause to THIS scenario's specific facts (concrete parties, amounts, "
        "dates, addresses) and to the REAL sources you retrieve. Fill or drop any "
        "{{placeholders}} the template may contain with the scenario facts.\n"
        "- The template is a drafting GUIDE, not the final document: do NOT paste its generic "
        "explanatory prose ('co to jest', 'jak wypełnić', 'częste błędy') verbatim into "
        "expected_output. Write a clean, professional document that uses the template's "
        "structure and substance.\n"
        "- Drop sections that do not apply and add any section the retrieved sources show is "
        "required. PRESERVE the CC-BY-4.0 source attribution that appears in the template "
        "result.\n\n"
        "WORKFLOW:\n"
        "0) TEMPLATE: call get_template with the doc_type (inferred from the user's request, "
        "e.g. a renting request -> umowa_najmu). Use it to plan the document outline.\n"
        "1) PLAN: reason about the drafting task; decide which legal areas/sections you "
        "need sources for and what to search.\n"
        "2) RETRIEVE: call keyword_search and/or semantic_search. You MAY issue several "
        "tool calls in ONE turn to gather evidence in parallel. Inspect the snippets.\n"
        "3) READ: call chunk_read on the most relevant chunk_ids to obtain full text "
        "before writing.\n"
        "4) WRITE: produce the COMPLETE scenario JSON (schema below). Ground "
        "expected_output in the retrieved sources AND the template, and cite articles by "
        "number (e.g. 'zgodnie z art. 355 KC ...').\n\n"
        "HARD CONSTRAINTS:\n"
        f"- {doc_type_hint}\n"
        "- The 'instruction' field MUST read like a LAYPERSON's ChatGPT message: "
        "first-person, casual, everyday Polish, NO legal jargon or article numbers.\n"
        "- expected_output is the COMPLETE drafted Polish legal document (full "
        "contract/letter, ALL standard sections), properly formatted, citing relevant "
        "articles by number. Do NOT truncate. Plan the outline first, then write the "
        "full document.\n"
        "- Cite only norms present in the sources you actually retrieved or in the fetched "
        "template.\n\n"
        "JSON schema to produce:\n"
        f"{schema}\n\n"
        "Output ONLY the final JSON object. No commentary, no markdown fences."
    )


def run_agentic(
    store,
    instruction: str,
    chat_main: Callable[..., Any],
    chat_tool: Optional[Callable[..., Any]] = None,
    max_iter: int = 6,
    read_set: Optional[set] = None,
    session_id: Optional[str] = None,
    doc_type: Optional[str] = None,
):
    """Run one agentic scenario generation.

    Returns a dict with ``messages`` (full native transcript), ``scenario``
    (parsed JSON or None), ``read_set`` (chunk_ids actually read) and ``ok``.
    """
    from legal_drafter import scenario_gen

    system = _build_system_prompt(scenario_gen._schema_skeleton(), scenario_gen.DOC_TYPE_HINT)
    messages: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": instruction},
    ]
    read_set = read_set if read_set is not None else set()
    last_parsed = None

    for _ in range(max_iter):
        last = messages[-1]
        use_tool = chat_tool if (chat_tool and last.get("role") == "tool") else chat_main
        resp = use_tool(messages, session_id=session_id)
        msg = resp.choices[0].message
        messages.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": _tc_to_dict(msg),
            }
        )
        if msg.tool_calls:
            tool_msgs = []
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except Exception:
                    args = {}
                if tc.function.name == "chunk_read" and "exclude_ids" not in args:
                    args["exclude_ids"] = sorted(read_set)
                if tc.function.name == "get_template" and not args.get("doc_type"):
                    args["doc_type"] = (doc_type or "other").strip().lower()
                result = execute_tool(store, tc.function.name, args)
                if tc.function.name == "chunk_read":
                    for r in result:
                        if r.get("chunk_id") and not r.get("already_read"):
                            read_set.add(str(r["chunk_id"]))
                tool_msgs.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tc.function.name,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )
            messages.extend(tool_msgs)
            continue
        last_parsed = _extract_json(msg.content or "")
        if last_parsed:
            return {
                "messages": messages,
                "scenario": last_parsed,
                "read_set": read_set,
                "ok": True,
            }
    return {"messages": messages, "scenario": last_parsed, "read_set": read_set, "ok": False}
