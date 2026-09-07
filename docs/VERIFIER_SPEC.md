# Verifier Specification

## 1. Purpose

The verifier checks whether generated legal clauses are grounded, safe, and consistent with retrieved legal evidence. It is separate from the drafting model.

The verifier checks whether generated legal clauses are grounded, safe, and consistent with retrieved legal evidence. It produces an evidence-backed audit trail for each clause verdict.

## 2. Problem this spec solves

The current prototype showed three failure modes:

1. **Prose-level NLI is insufficient**
   A model may treat a source about interest for late payment as support for a larger claim that also allows contractual penalties for late payment.

2. **Source polarity matters**
   A UOKiK abusive-clause entry can semantically match a drafted clause, but that match means risk, not support.

3. **Regex/rule patching does not scale**
   Hardcoded patterns can catch obvious traps but fail with negation, exceptions, and unseen legal issues.

The fix is to verify structured legal objects against source objects and use NLI only as one signal.

## 3. Verifier architecture

```text
draft section JSON
  -> clause segmentation
  -> legal-object extraction from draft
  -> evidence retrieval per object
  -> source-object extraction from evidence
  -> source semantics / polarity enrichment
  -> object comparison
  -> optional NLI relation check
  -> final verdict
  -> repair / warning / accept decision
```

## 4. Source taxonomy

Every evidence chunk must have a source type and semantics.

| source_type | authority_level | polarity | verifier meaning |
|---|---|---|---|
| `statute` | `binding_law` | `normative_rule` | Can support, limit, or contradict legal objects. |
| `abusive_clause` | `consumer_blacklist_or_registry` | `negative_example` | Similar draft wording is risky or likely non-binding. |
| `judgment` | `case_law` | `context_dependent` | Requires holding/reasoning extraction before use. |
| `official_form` | `official_form` | `form_requirement` | Supports required fields/layout, not broad legal claims. |
| `template` | `reviewed_template` | `drafting_example` | Supports drafting style/structure only. |
| `guide_or_playbook` | `secondary_guidance` | `guidance` | Supports risk checklists and explanations, weaker than primary law. |
| `unknown` | `unknown` | `unknown` | Cannot be used as strong support. |

### Source object fields

Every retrieved source chunk should expose:

```json
{
  "source_id": "sejm_eli_du_2024_1061_art_483",
  "source_type": "statute",
  "authority_level": "binding_law",
  "polarity": "normative_rule",
  "jurisdiction": "PL",
  "title": "Kodeks cywilny",
  "article_number": "483",
  "effective_date": null,
  "status": "current_or_consolidated",
  "url": "...",
  "text": "..."
}
```

## 5. Legal-object schema

The verifier should convert draft clauses and evidence into legal objects.

### 5.1 Core object

```json
{
  "object_id": "draft_sec5_clause3_obj1",
  "origin": "draft",
  "text_span": "Wykonawca może naliczyć karę umowną za opóźnienie w zapłacie.",
  "source_ids": [],
  "jurisdiction": "PL",
  "language": "pl",
  "actor": "Wykonawca",
  "counterparty": "Zamawiający",
  "modality": "permission",
  "action": "charge_contractual_penalty",
  "legal_object": "contractual_penalty",
  "trigger": "late_monetary_payment",
  "condition": null,
  "exception": null,
  "temporal_scope": null,
  "amount": null,
  "payment_kind": "monetary_payment",
  "obligation_kind": "monetary",
  "risk_category": "payment_penalty",
  "confidence": 0.82,
  "extraction_method": "model_or_rule",
  "span_start": null,
  "span_end": null
}
```

### 5.2 Field definitions

| Field | Meaning |
|---|---|
| `object_id` | Stable unique ID. |
| `origin` | `draft`, `source`, `user_fact`, or `playbook`. |
| `text_span` | Exact text supporting this object. |
| `source_ids` | Source IDs cited by draft or source object. |
| `jurisdiction` | Usually `PL`. |
| `actor` | Party bearing right/obligation/permission. |
| `counterparty` | Other party affected by the object. |
| `modality` | `obligation`, `permission`, `prohibition`, `right`, `risk_warning`, `definition`, `condition`, `exception`. |
| `action` | Normalized action such as `charge_interest`, `terminate_contract`, `change_price`. |
| `legal_object` | Thing or concept acted on, e.g. `contractual_penalty`, `deposit`, `court_jurisdiction`. |
| `trigger` | Event activating the rule, e.g. `late_payment`, `consumer_withdrawal`. |
| `condition` | Required condition for the action/rule. |
| `exception` | Exception or limitation. |
| `temporal_scope` | Deadline, notice period, duration, effective period. |
| `amount` | Money/percentage/fee if present. |
| `payment_kind` | `monetary_payment`, `non_monetary_performance`, `deposit`, `damages`, `interest`, etc. |
| `obligation_kind` | `monetary`, `non_monetary`, `mixed`, or `unknown`. |
| `risk_category` | Normalized risk class. |
| `confidence` | Extraction confidence. |
| `extraction_method` | `deterministic`, `small_model`, `llm_fallback`, or `human_label`. |

### 5.3 Modality values

| Modality | Meaning |
|---|---|
| `obligation` | Party must do something. |
| `permission` | Party may do something. |
| `prohibition` | Party must not do something. |
| `right` | Party has legal right/remedy. |
| `risk_warning` | Source warns a clause is risky or non-binding. |
| `definition` | Defines term or scope. |
| `condition` | Condition for another rule. |
| `exception` | Exception to another rule. |

### 5.4 Example source objects

A statute can produce a normative source object:

```json
{
  "object_id": "source_kc_483_obj1",
  "origin": "source",
  "source_ids": ["sejm_eli_du_2024_1061_art_483"],
  "source_type": "statute",
  "authority_level": "binding_law",
  "polarity": "normative_rule",
  "modality": "permission",
  "action": "reserve_contractual_penalty",
  "legal_object": "contractual_penalty",
  "trigger": "non_monetary_obligation_nonperformance_or_improper_performance",
  "obligation_kind": "non_monetary"
}
```

An abusive-clause registry entry can produce a negative example object:

```json
{
  "object_id": "source_uokik_300_obj1",
  "origin": "source",
  "source_ids": ["uokik_abusive_clause_300"],
  "source_type": "abusive_clause",
  "authority_level": "consumer_blacklist_or_registry",
  "polarity": "negative_example",
  "modality": "risk_warning",
  "action": "treat_silence_as_acceptance",
  "trigger": "material_contract_change",
  "risk_category": "silence_as_acceptance"
}
```

## 6. Verification pipeline

### 6.1 Step 1: extract draft legal objects

Input:
- `sections_pl.json`.

Output:
- List of draft legal objects.

Extraction methods:
1. Deterministic segmentation by clause/sentence.
2. Small local/CPU model extraction where possible.
3. LLM fallback only for ambiguous high-risk clauses.
4. Human labels for evaluation data.

The verifier must not rely only on raw clause prose.

### 6.2 Step 2: retrieve evidence per object

Evidence query should use object fields, not only full clause text.

Example:

```json
{
  "action": "charge_contractual_penalty",
  "trigger": "late_monetary_payment",
  "obligation_kind": "monetary",
  "risk_category": "payment_penalty"
}
```

Retrieval should return:
- Exact cited sources from draft.
- Statutes matching normalized action/trigger.
- Negative-example sources matching risk category.
- Judgments only when relevant and with holding/reasoning extraction.

### 6.3 Step 3: extract source legal objects

For every top evidence source, extract source objects with the same schema.

Source extraction must preserve:
- Conditions.
- Exceptions.
- Negative polarity.
- Monetary/non-monetary distinction.
- Effective status/date.
- Jurisdiction.

### 6.4 Step 4: compare objects

Object comparison is the primary verifier.

Checks:

| Check | Example |
|---|---|
| Action match | `charge_contractual_penalty` vs `reserve_contractual_penalty`. |
| Trigger match | `late_monetary_payment` vs `non_monetary_obligation`. |
| Modality match | `permission` vs `prohibition` or `risk_warning`. |
| Condition coverage | Draft includes required written notice/serious breach condition. |
| Exception coverage | Draft does not omit exception that limits the rule. |
| Polarity | Negative example means similar draft object is risky. |
| Authority | Binding statute outranks template. |
| Conflict | One source supports part of claim, another contradicts another part. |

### 6.5 Step 5: optional NLI check

NLI remains useful, but secondary.

Use NLI to compare:
- Draft text span vs evidence span.
- Source object explanation vs draft object explanation.

NLI labels:
- `entailment`.
- `contradiction`.
- `neutral`.

NLI must not override object mismatch. If NLI says a source about interest entails a claim that also allows contractual penalties, object comparison should still reject the penalty part.

### 6.6 Step 6: source-polarity decision

Decision examples:

| Object comparison | Source polarity | Final verdict |
|---|---|---|
| Draft object matches statute permission/obligation with all conditions | `normative_rule` | `supported` |
| Draft object omits statutory condition/exception | `normative_rule` | `contradicted` or `insufficient` |
| Draft object matches UOKiK abusive clause | `negative_example` | `risky` |
| Draft object explicitly avoids UOKiK abusive pattern | `negative_example` | `supported` or `risk_reduced` |
| Draft object only matches template | `drafting_example` | `insufficient` or `weak_support` |
| Draft object matches judgment but holding is unclear | `context_dependent` | `needs_review` |

## 7. Verdict model

### 7.1 Final verdict labels

| Verdict | Meaning | Runtime action |
|---|---|---|
| `supported` | Evidence supports the object and required conditions are covered. | Accept. |
| `risky` | Object matches negative source, risky playbook pattern, or consumer-abusive signal. | Repair or warn. |
| `contradicted` | Binding/normative source contradicts the object. | Repair/block. |
| `insufficient` | Evidence does not support or refute the object. | Warn, ask for more sources, or mark for review. |
| `needs_review` | Case-law conflict, low confidence, or high-stakes ambiguity. | Human review. |

### 7.2 Confidence

Each verdict should include:

```json
{
  "verdict": "risky",
  "confidence": 0.86,
  "primary_reason": "draft_object_matches_negative_example_source",
  "object_ids": ["draft_5_3_obj1"],
  "evidence_ids": ["uokik_abusive_clause_300"],
  "nli_relation": "entailment",
  "object_relation": "same_action_same_trigger",
  "source_polarity": "negative_example"
}
```

## 8. Repair policy

Repair should be object-targeted.

1. If verdict is `supported`, do nothing.
2. If verdict is `risky` or `contradicted`, send only the affected clause/object to repair.
3. If verdict is `insufficient`, either add a warning or request more evidence.
4. If verdict is `needs_review`, keep draft but mark review required.
5. Allow only one repair attempt per section.
6. Never retry the whole run automatically.

Repair prompt should include:
- Original clause.
- Failing legal object.
- Verdict.
- Evidence source IDs.
- Safe object pattern if available.
- Instruction to preserve non-failing facts.

## 9. Evaluation datasets

### 9.1 Current datasets

Current seed labels:
- `eval/legal_drafting_rag_cases.jsonl`: five source-backed drafting trap cases.
- `eval/verifier_claim_labels.jsonl`: ten claim-level verifier labels.

Current verifier baseline:
- CPU Modal full run reached 9/10 on the ten labels using reranker + NLI + source polarity.
- The failure was a compound claim where one evidence source supported interest for late payment but not contractual penalty for late payment.
- This failure motivates object decomposition, not a single hardcoded payment rule.

### 9.2 Required next labels

Before implementation, add labels for legal objects, not only prose claims.

Each label should include:

```json
{
  "label_id": "example_001",
  "case_id": "...",
  "input_text": "...",
  "expected_objects": [
    {
      "modality": "permission",
      "action": "charge_contractual_penalty",
      "trigger": "late_monetary_payment",
      "obligation_kind": "monetary",
      "expected_verdict": "risky"
    }
  ],
  "expected_evidence_ids": ["sejm_eli_du_2024_1061_art_483"],
  "human_review_status": "pending"
}
```

### 9.3 Split policy

Use separate splits:

| Split | Purpose |
|---|---|
| `dev` | Used while designing extraction/comparison logic. |
| `test` | Used after implementation to check quality. |
| `heldout` | Not inspected until milestone review. |

A fix is not accepted if it only improves `dev` and harms `test`/`heldout`.

## 10. Metrics

Track metrics separately.

| Metric | Target |
|---|---|
| Retrieval recall@k for expected evidence | High before verifier runs. |
| Evidence precision@k | Improve with reranking. |
| Legal-object extraction F1 | Main metric for object layer. |
| Source-polarity accuracy | Must distinguish positive law vs negative examples. |
| Verifier verdict accuracy | Supported/risky/contradicted/insufficient/needs_review. |
| False-safe rate | Highest priority to minimize. |
| False-block rate | Important, but less dangerous than false-safe. |
| Repair success rate | Measure after verifier is stable. |
| Human reviewer accept/edit/reject rate | Final quality metric. |

## 11. Acceptance criteria for next milestone

Before more GPU drafting runs, the verifier milestone should satisfy:

1. Legal-object schema implemented in code.
2. At least 20 labeled object examples.
3. At least 5 held-out object examples.
4. Retrieval eval passes expected evidence checks for the labels.
5. Object extraction and verifier eval run locally or CPU Modal.
6. Verifier report separates:
   - extraction error,
   - retrieval error,
   - source-polarity error,
   - object-comparison error,
   - NLI error.
7. No logic change is accepted without updating labels and reports.

## 12. Implementation sequence

### Step 1: labels

Create `eval/verifier_object_labels.jsonl` from current ten claim labels.

Each label should identify:
- atomic legal objects,
- expected verdict per object,
- expected source IDs,
- source polarity,
- whether the object is safe/risky.

### Step 2: object extraction

Create `src/legal_drafter/legal_objects.py`.

Responsibilities:
- Extract objects from draft clauses.
- Extract objects from source text where possible.
- Normalize actors/actions/triggers.
- Preserve original text spans.

### Step 3: object evidence selection

Extend RAG query construction to use object fields.

Example query components:
- action.
- trigger.
- risk category.
- obligation kind.
- source type preference.

### Step 4: object comparator

Create `src/legal_drafter/object_verifier.py`.

Responsibilities:
- Compare draft object to source objects.
- Apply source polarity.
- Use NLI only as secondary signal.
- Return structured verdict.

### Step 5: reporting

Create reports showing:
- Draft object.
- Evidence objects.
- Source IDs.
- Verdict.
- Reason.
- Confidence.
- Error category.

### Step 6: integration with drafting

Only after object verifier passes CPU eval:
- Integrate verifier into `src/legal_drafter/pipeline.py` after section generation.
- One targeted repair attempt for failed clauses/objects.
- Do not regenerate whole document.

## 13. Anti-pattern rules

Do not implement a new verifier rule unless it maps to one of:

- A legal-object schema field.
- A source taxonomy rule.
- A source-polarity rule.
- A general object-comparison rule.
- A labeled evaluation gap.

Bad approach:

```text
if current_case_id == "real_kc_payment_delay_penalty": fail
```

Avoid narrow pattern fixes such as:

```text
if text contains "art. 483": fail payment penalty
```

Preferred approach:

```text
extract object:
  action = charge_contractual_penalty
  trigger = late_monetary_payment
  obligation_kind = monetary
compare against source object:
  action = reserve_contractual_penalty
  trigger = non_monetary_nonperformance
  obligation_kind = non_monetary
verdict = contradicted/risky because trigger and obligation_kind mismatch
```

## 14. Relationship to existing code

Existing code remains useful:

| Existing module | Future role |
|---|---|
| `src/legal_drafter/rag.py` | Evidence retrieval backend. |
| `src/legal_drafter/source_semantics.py` | Source taxonomy and polarity enrichment. |
| `src/legal_drafter/claim_extraction.py` | Temporary claim splitter; can feed object extraction. |
| `src/legal_drafter/evidence_reranker.py` | Candidate evidence selection. |
| `src/legal_drafter/nli_verifier.py` | Secondary NLI check. |
| `src/legal_drafter/legal_constraints.py` | Draft-time hints; not final verifier. |
| `src/legal_drafter/legal_warnings.py` | Cheap guardrail/triage; not final legal reasoning. |

Next milestone: legal-object verifier (Section 12).
