# Schema Specification — Polish Legal Drafter (KAFT distillation)

> Canonical reference for the three data objects used in the distillation
> pipeline, plus the on-disk training format. **Sections marked
> `[OBSERVED]` already exist in `task_pool_v1.jsonl`; `[PROPOSED]` are
> decisions to confirm before generation begins.**
>
> Status: the *question* (task) and *grounding* (evidence_report) layers
> are real and mostly complete; the *answer* (draft) layer is missing and
> must be defined + generated. The 117 tasks lacking an `evidence_report`
> must be regenerated before they can be used.

---

## 1. Task (question / input) — `[OBSERVED]`

A single drafting request. Top-level keys (example `schema_version =
distill_drafting_task_v1`):

```json
{
  "task_id": "distill_pilot_v1_0001_contractual_penalty_non_monetary",
  "schema_version": "distill_drafting_task_v1",
  "topic": "contractual_penalty_non_monetary",
  "instruction": "<natural-language drafting instruction, Polish>",
  "facts": {
    "doc_type": "UMOWA_ZLECENIA",
    "doc_title_hint": "Umowa zlecenia",
    "party_a": "Zleceniodawca",
    "party_b": "Zleceniobiorca",
    "subject": "obsługa rezerwacji i komunikacji z klientem",
    "amount": "[[REMUNERATION_AMOUNT]]",
    "payment_deadline": "[[PAYMENT_DEADLINE_DAYS]]",
    "notice_days": "[[NOTICE_DAYS]]",
    "topic_focus": "contractual_penalty_non_monetary",
    "scenario_variant": "<free-text variant note>"
  },
  "source_refs": ["sejm_eli:sejm_eli_du_2024_1061_art_483", "..."],
  "sources": [ { /* see below */ } ],
  "evidence_report": { /* see §2 */ },
  "expected_output": "structured Polish draft sections with source IDs and risk notes"
}
```

### `facts` fields
`doc_type`, `doc_title_hint`, `party_a`, `party_b`, `subject`, `amount`,
`payment_deadline`, `notice_days`, `topic_focus`, `scenario_variant`.
Variable slots use `[[PLACEHOLDER]]` tokens (observed: `REMUNERATION_AMOUNT`,
`REMUNERATION_CURRENCY`, `PAYMENT_DEADLINE_DAYS`, `NOTICE_DAYS`,
`RENT_AMOUNT/RENT_CURRENCY`, `PRICE_AMOUNT/PRICE_CURRENCY`).

### `sources[]` fields (retrieved RAG context, one per statute chunk)
```json
{
  "source_ref": "sejm_eli:sejm_eli_du_2024_1061_art_483",
  "source_type": "statute",
  "publisher": "Sejm ELI",
  "title": "Obwieszczenie ... Kodeks cywilny",
  "display_address": "Dz.U. 2024 poz. 1061",
  "article_number": "483",
  "top_category": null,
  "risk_or_safe": null,
  "regime_role": null,
  "evidence_role": "operative_required",
  "norm_ids": ["KC:483"],
  "act_id": "KC",
  "legal_issue_ids": ["contractual_penalty_non_monetary"],
  "text": "<verbatim statute chunk>"
}
```
Each source carries a **stable `source_ref`** and **`norm_ids`** (e.g.
`KC:483`, `TENANT_ACT:6`) — these are the citation anchors used by the
answer layer and the grounding layer.

**Assessment:** this layer is *better than* the RAFT/KAFT standard (it
already bundles instruction + ID'd, article-referenced sources). ✅

---

## 2. Evidence report (grounding layer) — `[OBSERVED]`

Pre-computed citation plan linking the task to the legal norms that support
it. This is the RAFT "oracle selection" made explicit. **Full schema
(verified on 2026-08-22):**

```json
{
  "task_id": "distill_pilot_v1_0001_contractual_penalty_non_monetary",
  "topic": "contractual_penalty_non_monetary",
  "regime": {
    "primary_regime": "zlecenie_general",
    "doc_type": "UMOWA_ZLECENIA",
    "topic": "contractual_penalty_non_monetary",
    "confidence": "medium",
    "required_source_roles": ["applicable_rule"],
    "forbidden_regimes": [],
    "must_use_concepts": [],
    "forbidden_concepts": [],
    "notes": []
  },
  "legal_issues": [
    {
      "issue_id": "contractual_penalty_non_monetary",
      "clause_functions": ["non_monetary_obligation", "penalty_amount", "mitigation_and_excess"],
      "must_support": ["KC:483", "KC:484"],
      "may_support": ["KC:471", "KC:472", "KC:473"],
      "allowed_acts": ["KC"]
    }
  ],
  "required_norms": ["KC:483", "KC:484"],
  "covered_required_norms": ["KC:483", "KC:484"],
  "missing_required_norms": [],
  "selected_norms": ["KC:471", "KC:472", "KC:473", "KC:483", "KC:484"],
  "selected_count": 6,
  "eligible_count": 5,
  "negative_count": 6552
}
```

**Field rules (to enforce on generation):**
- `regime.doc_type` MUST equal `facts.doc_type`.
- `selected_norms` / `required_norms` MUST be a subset of the task's own
  `sources[].norm_ids` (otherwise they don't resolve).
- `legal_issues[].must_support` / `may_support` MUST be `norm_ids` present
  in the task's sources.
- `covered_required_norms` ⊆ `required_norms`; `missing_required_norms`
  = `required_norms` − `covered_required_norms`.
- `notes` is a **list** (not a string).
- `confidence` ∈ {high, medium, low}.

**Coverage gap:** 153 tasks in `task_pool_v1.jsonl` lack a usable
`evidence_report`:
- **117** are nowhere covered (no `evidence_report` in any `_evidence` file)
  → require LLM regeneration.
- **36** are recoverable without LLM (16 cloneable from a present sibling
  of the same `doc_type`+`topic_focus`; 20 found in other `_evidence`
  files).

**Assessment:** this layer *exceeds* the RAFT standard (explicit
issue→norm mapping). ✅ for the 247 present; ⚠️ for the 153 missing.

---

## 3. Draft / answer (output) — `[PROPOSED — TO CONFIRM]`

> **This layer does not exist yet.** No draft documents are present in the
> repo; `expected_output` is only a free-text description. This is the
> single hard blocker for distillation.

Proposed shape (a grounded Polish legal document):

```json
{
  "draft": "<full Polish document text, with inline [source_ref] markers>",
  "citations": [
    {
      "marker": "[sejm_eli:sejm_eli_du_2024_1061_art_483]",
      "source_ref": "sejm_eli:sejm_eli_du_2024_1061_art_483",
      "norm_id": "KC:483",
      "quote": "<verbatim substring of the cited source.text>"
    }
  ],
  "reasoning": "<optional Chain-of-Thought: why each clause is grounded>"
}
```

**Citation convention (proposed):**
- Every factual/legal claim in `draft` carries an inline marker
  `[source_ref]` (the grounding contract — uncited claims are rejected,
  per the `legal-rag-pipeline` pattern).
- `citations[].quote` MUST be a verbatim substring of the referenced
  `sources[].text` (whitespace/case-normalized).
- `norm_id` references the statute (`KC:483`), `source_ref` the chunk.
- `reasoning` is optional but recommended (RAFT/KAFT show CoT in the
  target improves fidelity).

Open question: draft as **free text + markers** (above) vs **structured
sections** (`{sections:[{heading, body}]}`). Free text is simpler for
training (assistant turn = text); recommend free text + a side `citations`
list.

---

## 4. Distillation training format — `[PROPOSED — TO CONFIRM]`

Canonical SFT/distillation format is **ChatML `messages`** (what
HuggingFace / Unsloth / TRL / OpenAI / LLaMA-Factory all default to).
Each row = one JSONL object:

```json
{
  "messages": [
    {"role": "system", "content": "You are a Polish legal drafting expert. Draft the document grounded in the retrieved sources; cite every claim with [source_ref] and a verbatim quote."},
    {"role": "user",   "content": "<instruction> \n\n<facts with placeholders filled> \n\nRETRIEVED SOURCES:\n[source_ref=...|norm_id=KC:483|art.483] <verbatim text>\n...\n\nDISTRACTORS:\n<irrelevant chunks>"},
    {"role": "assistant", "content": "<draft with [source_ref] citations + optional CoT>"}
  ],
  "metadata": {
    "task_id": "distill_pilot_v1_0001_contractual_penalty_non_monetary",
    "schema_version": "distill_drafting_task_v1",
    "doc_type": "UMOWA_ZLECENIA",
    "topic": "contractual_penalty_non_monetary",
    "selected_norms": ["KC:483", "KC:484"],
    "evidence_report": { /* §2 */ },
    "retrieved_source_refs": ["..."],
    "draft_citations": [ /* §3 citations */ ]
  }
}
```

**Rules:**
- `metadata` is ignored by the trainer and used only for eval/grounding
  checks — it carries the `evidence_report` and citation map so the
  grounding layer survives the conversion.
- RAFT packing: include the oracle sources + a few distractor sources in
  the `user` turn; for a fraction `P` of rows, drop the oracle (forces
  memorization fallback). Recommended `P ≈ 0.8`.
- The `assistant` turn is the teacher-generated draft (response-based /
  offline distillation). Generate the draft with a rich CoT+citation
  template, but train the student on the simple prompt it will see at
  inference (context distillation).

---

## 5. Build checklist (what must happen before training)

| Step | Layer | Status |
|---|---|---|
| Define draft/answer schema (§3) | answer | ❌ proposed, confirm |
| Generate drafts for tasks | answer | ❌ not started |
| Regenerate 117 `evidence_report`s (full §2) | grounding | ❌ needs LLM |
| Merge/clone 36 recoverable `evidence_report`s | grounding | ❌ no LLM needed |
| Wrap tasks → ChatML `messages`+`metadata` (§4) | format | ❌ converter needed |

---

## References (provenance)
- Zhang et al., *RAFT: Adapting Language Model to Domain Specific RAG*, arXiv:2403.10131 — https://arxiv.org/abs/2403.10131
- Cai et al., *Knowledge Augmented Finetuning Matters in both RAG and Agent Based Dialog Systems*, arXiv:2506.22852 — https://arxiv.org/html/2506.22852
- Zhong et al., *KaFT: Knowledge-aware Fine-tuning…*, Findings ACL 2025, arXiv:2505.15480 — https://aclanthology.org/2025.findings-acl.1235/
- Shirgaonkar et al., *Knowledge Distillation Using Frontier Open-Source LLMs*, arXiv:2410.18588 — https://arxiv.org/html/2410.18588
- Unsloth, *Datasets Guide* (ChatML default) — https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide
- ehzawad/*legal-rag-pipeline* (grounding contract: `[evidence_id]` + verbatim quote) — https://github.com/ehzawad/legal-rag-pipeline
