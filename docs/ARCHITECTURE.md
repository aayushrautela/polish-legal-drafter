# Architecture

## 1. Purpose

This project is a thesis-grade Polish legal-document drafting assistant. It should produce structured legal drafts from user facts, retrieve trusted legal sources, detect risky or unsupported legal content, and give a human reviewer a clear evidence trail.

This document defines the target architecture for the system.

## 2. Product boundary

### Runtime user flow

```text
user request / uploaded facts
  -> document-type classification
  -> intake schema and missing-question loop
  -> structured facts JSON
  -> source retrieval and evidence pack
  -> legal-object constraints
  -> Qwen drafting model
  -> independent verifier
  -> repair or warning gate
  -> deterministic rendering
  -> draft + evidence + warnings for user/human review
```

The human reviewer does not check the input before every model run. The system should run automatically and output a draft. Human review is required before legal use and is also used during development to label/evaluate examples.

### In scope

- Polish legal-document drafting.
- Multiple document types over time: `UMOWA_ZLECENIA`, `UMOWA_NAJMU`, notices/replies, forms, annexes, and other common civil-law documents.
- Retrieval over trusted Polish legal sources.
- Structured JSON drafting and deterministic rendering.
- Independent legal grounding/risk verification.
- Evaluation harness for thesis comparison.

### Out of scope

- Fully autonomous legal advice.
- Treating LLM output as authoritative law.
- Relying only on one model to draft and self-certify correctness.
- Per-case regex fixes as the primary validation strategy.
- Runtime dependency on paid external legal-data APIs as the main source of law.

## 3. Design principles

1. **Compound system, not one chatbot**
   Compound-system design: retrieval, source metadata, clause segmentation, classification, entity extraction, risk scoring, generation, and audit logs.

2. **Primary sources first**
   Statutes, official forms, UOKiK records, judgments, and reviewed templates must be stored with source IDs, dates, jurisdiction, status, source type, and provenance.

3. **Source meaning matters**
   A retrieved source can be positive law, a negative example, a judgment, a template, or guidance. Entailment alone is insufficient. For example, a UOKiK abusive-clause registry entry is evidence that a similar clause is risky, not evidence that it should be drafted.

4. **Verify structured legal objects, not only prose**
   Legal clauses should be decomposed into structured objects: actor, modality, action, trigger, condition, exception, temporal scope, amount, and source basis. This prevents failures where a model treats one part of a complex sentence as support for the whole claim.

5. **Independent verifier**
   Qwen drafts. A separate verifier stack checks claims/objects against evidence. The verifier may use a reranker, NLI model, legal-object comparison, and source-polarity logic. Qwen can be used as an optional critic, but it should not be the final authority on its own draft.

6. **Human review is an output-stage requirement**
   The system can automatically draft and verify, but final legal use needs human review. During development, human-labeled examples are used to test and improve the verifier.

7. **No per-case tuning rule**
   A change is not accepted just because it fixes one failing example. Each verifier change must be tied to a general rule or schema change and tested against existing plus held-out examples.

8. **Evaluation gates before expansion**
   Add document types, larger corpora, and model-training releases only after the current architecture passes retrieval, verifier, and drafting evaluation.

## 4. Current implementation map

Current prototype:

| Layer | Current files | Current status |
|---|---|---|
| Drafting pipeline | `src/legal_drafter/pipeline.py` | Section-by-section JSON generation, repair, retrieval injection, rendering. |
| Document specs | `config/document_specs/umowa_zlecenia.v1.json` | One public supported document type; more require matching specs/locales/tests. |
| Prompt templates | `config/prompts/*.txt` | Externalized draft/repair prompts. |
| Rendering | `src/legal_drafter/renderer.py`, `config/locales/*.json` | Deterministic Markdown rendering. |
| Retrieval | `src/legal_drafter/rag.py` | Local sparse retrieval, source boosts, section filters, required-source injection. |
| Constraints | `src/legal_drafter/legal_constraints.py` | Rule-derived constraints from facts + retrieval. Useful but not sufficient as final architecture. |
| Warnings | `src/legal_drafter/legal_warnings.py` | Current risk/blocker checks. Should become secondary guardrails, not core legal reasoning. |
| Verifier scaffold | `src/legal_drafter/claim_extraction.py`, `src/legal_drafter/evidence_reranker.py`, `src/legal_drafter/nli_verifier.py`, `src/legal_drafter/source_semantics.py` | Independent reranker/NLI/source-polarity verifier prototype. |

Current constraints/warnings are not sufficient for final legal reasoning. The next step is to formalize the verifier and legal-object layer.

## 5. Target architecture

### 5.1 Intake and document-type layer

Responsible for turning user language into structured facts.

Inputs:
- User request.
- Optional uploaded notice/document.
- Jurisdiction and language.
- Target document type.

Outputs:
- `doc_type`.
- `legal_area`.
- `structured_facts`.
- `missing_questions`.
- `risk_hints` from user facts.

Required capabilities:
- Classify request into supported document type.
- Ask missing questions using document-specific schema.
- Preserve user intent without accepting legally risky instructions as valid.
- Mark risky user requests separately from safe facts.

### 5.2 Source ingestion and provenance layer

Responsible for trusted legal corpus creation.

Source classes:

| Source class | Examples | Role |
|---|---|---|
| `statute` | Sejm ELI acts | Binding or normative legal basis. |
| `abusive_clause` | UOKiK registry | Negative examples / risk signals. |
| `judgment` | SAOS | Context-dependent case law; requires holding extraction. |
| `official_form` | Ministry of Justice forms | Required fields/layouts. |
| `template` | Reviewed contract templates | Drafting structure/style, not law by itself. |
| `guide_or_playbook` | Reviewed legal guide, internal playbook | Practical checklist/risk rules. |

Every source must store:
- `source_id`.
- `source_type`.
- Title.
- Publisher.
- Jurisdiction.
- Date/status/effective date if available.
- URL/path/checksum.
- License/terms if relevant.
- Review status.
- Source polarity and authority level.

### 5.3 Retrieval layer

Retrieval should produce an evidence pack, not just random chunks.

Minimum retrieval pipeline:

```text
query / legal object / section need
  -> metadata filter by jurisdiction, source_type, legal_area, doc_type
  -> sparse retrieval for exact legal terms/articles
  -> dense retrieval or embedding retrieval
  -> cross-encoder reranking
  -> source-polarity enrichment
  -> evidence pack with source IDs and previews
```

Current sparse retrieval is acceptable for pilot work, but production/thesis-quality RAG should add:
- Hybrid retrieval: sparse + embeddings.
- Cross-encoder reranker.
- Metadata filters before retrieval.
- Parent-child source context.
- Source freshness/status checks.
- Retrieval eval with expected sources.

### 5.4 Source semantics layer

This layer interprets what a source means.

Examples:

| Source type | Polarity | Meaning |
|---|---|---|
| Statute | `normative_rule` | Source can support, limit, or contradict a legal object. |
| UOKiK abusive clause | `negative_example` | A similar clause is a risk/prohibition signal. |
| Judgment | `context_dependent` | Needs holding/reasoning extraction before use. |
| Official form | `form_requirement` | Supports required fields/layouts. |
| Template | `drafting_example` | Supports style/structure, not legal correctness alone. |

The verifier must not treat all retrieved text as positive support.

### 5.5 Legal-object layer

Legal text should be converted into structured objects before final verification.

Example:

```json
{
  "object_id": "draft_6_3_obj_1",
  "source_text": "Wykonawca może naliczyć karę umowną za opóźnienie w zapłacie.",
  "actor": "Wykonawca",
  "counterparty": "Zamawiający",
  "modality": "permission",
  "action": "charge_contractual_penalty",
  "trigger": "late_monetary_payment",
  "obligation_kind": "monetary",
  "condition": null,
  "exception": null,
  "temporal_scope": null,
  "amount": null,
  "source_ids": [],
  "confidence": 0.82
}
```

The same representation should be extracted from relevant source chunks where possible.

References:
- Akoma Ntoso for legal document structure and metadata.
- LegalRuleML for deontic/legal norms, temporal scope, defeasibility, and source links.
- ContractNLI for evidence-span-backed legal entailment.
- Legal graph/RAG systems for entity-relation grounding and audit trails.

### 5.6 Drafter layer

Qwen remains the drafting model.

Responsibilities:
- Generate structured section JSON.
- Respect intake facts, document schema, and evidence pack.
- Cite only retrieved source IDs.
- Avoid drafting prohibited user requests when constraints say they are risky.

Qwen should not be responsible for final legal verification of its own output.

### 5.7 Verifier layer

Verifier responsibilities:

1. Extract legal claims/objects from draft clauses.
2. Select evidence for each claim/object.
3. Interpret source semantics.
4. Compare draft legal objects against source legal objects.
5. Use NLI/entailment as secondary support, not the only decision.
6. Return structured verdicts.

Verifier outputs:

| Verdict | Meaning | Runtime action |
|---|---|---|
| `supported` | Evidence supports the legal object. | Accept. |
| `risky` | Similar to negative source or violates source meaning. | Repair or warn. |
| `contradicted` | Binding/normative source contradicts the claim. | Repair or block. |
| `insufficient` | Evidence is missing or ambiguous. | Add warning/source request. |
| `needs_review` | Case law/complex conflict/low confidence. | Mark for human review. |

### 5.8 Repair and warning gate

The repair gate should not endlessly regenerate.

Policy:
- One repair attempt per section for prompt/model failure.
- No whole-run retry.
- If still failed after one repair, output failure/warning metadata.
- High-risk legal blockers should not be hidden.
- Repair should target specific legal objects/clauses, not regenerate blindly.

### 5.9 Rendering layer

Rendering should remain deterministic.

The model produces structured JSON. Code renders:
- Headings.
- Clause numbering.
- Parties/signature blocks.
- Safety/human-review notice.
- Source/warning appendix.

### 5.10 Audit and provenance layer

Every run should be reproducible.

Run folder should include:
- Input facts.
- Model/version/GPU/engine settings.
- Retrieved sources.
- Legal objects extracted from draft and sources.
- Verifier verdicts.
- Repairs attempted.
- Final draft.
- Report metrics.
- Timestamped output path.

This supports thesis evaluation and expert review.

## 6. Evaluation and TEVV

TEVV means test, evaluation, verification, and validation.

The project needs separate evaluations for:

| Evaluation | Question answered |
|---|---|
| Retrieval eval | Did we retrieve the right sources? |
| Source semantics eval | Did we interpret source polarity correctly? |
| Legal-object extraction eval | Did we extract actor/action/modality/condition correctly? |
| Verifier eval | Did we classify supported/risky/contradicted/insufficient correctly? |
| Drafting eval | Is the final document useful and structured? |
| Human review eval | Would a legal reviewer accept, edit, or reject the clause? |

Current state:
- Retrieval pilot passes 5/5 targeted cases.
- Verifier label pilot reached 9/10 with source polarity + NLI, but failed on a compound claim where one source supported only part of the claim.
- This motivates legal-object decomposition rather than more per-case patches.

## 7. Development policy from now on

1. **Spec first**: update `VERIFIER_SPEC.md` before changing verifier logic.
2. **Labels before fixes**: add or update labels before implementing a new behavior.
3. **Held-out cases**: keep some examples unseen until after implementation.
4. **No per-case patching**: every fix must generalize to a schema, source-semantics, extraction, or verifier rule.
5. **No unnecessary GPU runs**: local/CPU validation first; GPU only after explicit approval.
6. **Separate metrics**: do not treat final draft success as proof that retrieval/verifier is correct.
7. **Human review required**: final legal use remains human-reviewed.

## 8. Migration plan

### Phase 1: Freeze current baseline

- Keep current Qwen/vLLM drafting pipeline.
- Keep current RAG and source-polarity verifier as baseline.
- Save current 5-case drafting eval and 10-label verifier eval as baseline results.

### Phase 2: Legal-object verifier

- Define legal-object schema in `VERIFIER_SPEC.md`.
- Create labeled legal-object examples for existing 10 verifier labels.
- Implement deterministic + model-assisted object extraction.
- Compare draft objects to source objects.
- Use NLI only as secondary evidence.

### Phase 3: Better retrieval

- Add embedding retrieval and cross-encoder reranking to RAG.
- Store source metadata in a persistent vector/search database.
- Expand retrieval eval to 20-50 cases per document type.

### Phase 4: Document-type expansion

- Add `UMOWA_NAJMU` because tenant/rental sources and risk cases are already available.
- Then add notices/replies and other document types.

### Phase 5: Model comparison and training release

Only after schemas, verifier, retrieval, and evaluation are stable:
- Prepare human-reviewed `facts -> draft JSON` training examples.
- Compare model-serving and training variants in a separate reproducible release.
- Keep verifier independent from drafter.

## 9. Success criteria

The architecture is working when:

- Draft JSON is structurally valid.
- Relevant sources are retrieved and cited.
- Negative sources such as UOKiK are treated as risk evidence, not drafting support.
- Compound claims are decomposed before verification.
- Verifier distinguishes supported, risky, contradicted, insufficient, and needs-review cases.
- Output includes an audit trail a reviewer can inspect.
-    Improvements are measured on held-out examples, not only on single failing cases.
