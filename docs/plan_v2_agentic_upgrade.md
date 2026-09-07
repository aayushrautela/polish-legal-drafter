# PLAN v2 — Agentic curation + verified questions (NOT yet implemented)

Status: APPROVED DIRECTION, awaiting green light to build. Nothing here changes
the frozen v266 zlecenia set or the running batch-1 phase3 — this is the
**next-generation pipeline** for future doc types and future rounds.

User's original vision, validated against 2025–26 research (diary #92):
> AI takes the template → uses RAG tools to pull what IT needs (no force-feeding)
> → judges relevance → add/remove/change → saves an extended template + the extra
> RAG items as an artifact → next call generates a question from template+extras
> → next call generates the answer.
>
> Goal: every question legally sound AND provably answerable from our own RAG data.

Research backing: RAGShaper 2601.08699 (InfoCurator explores KB before any Q&A);
KARL 2603.05218 (tool-exploring QA synthesizer + dedup agent + solver pass-rate
band filter); SAGE EACL-F2026 (generator↔solver loop, execution feedback,
pass@K filter); S³-R1 2605.01248 (oracle-vs-retrieved F1 answerability gate);
CRAG 2401.15884 (+reproduction 2603.16169 caveat: judges are entity-matchers,
need few-shot); Self-RAG 2310.11511 (IsRel tokens); German-law FT 2601.14160
(statute-only grounding rule; naive one-prompt generation DEGRADED models);
LawGPT/KGDG 2502.06572 (fixer + derivable-from-refs verifier).

## Pipeline v2 (insertions marked ⊕; existing phases unchanged)

Phase A  corpus growth            — unchanged (see recipe.md)
Phase B  candidate sweep          — unchanged deterministic pre-filter
                                    (embeddings propose, gates dispose)
⊕ PHASE BETA — EXPLORER-JUDGE       — NEW, replaces pure agent-curation gap:
    1. Give teacher-model: doc_type template + tool access
       (semantic_search / chunk_read via existing execute_tool).
    2. Model pulls candidates IT decides it needs (bounded max_iter ~6),
       for each emits CRAG-style verdict keep/drop + one-line reason +
       intended role (statute base / clause add / red flag).
    3. Deterministic post-filter still applies (id allowlists, length,
       dedup) — model judgment does NOT bypass mechanical hygiene.
    4. Output artifact: `extended_template.jsonl` per doc type =
       template + accepted extras + judge reasons (auditable, frozen w/ sha).
    Few-shot REQUIREMENT: judge prompts must include ≥2 worked examples
    (Italian Civil Code study: 0-shot judges mass-false-negative;
     2-shot ≈ human-level).
Phase C  freeze                   — unchanged, now freezes extended_template too
Phase D  phase2 mutations         — unchanged batch-1 config
Phase E  phase3 questions         — unchanged batch-1 config
⊕ Phase E+ SOLVE-BACK GATE        — NEW, makes "answerable by RAG" provable:
    For each generated question, teacher answers it using ONLY that anchor's
    frozen material (template + accepted extras). Keep question iff
    answer consistent w/ the mutations the question was derived from
    (S³-R1 oracle-vs-retrieved pattern; KGDG DAVER "derivable from refs").
    Reuse repo `nli_verifier.py` for entailment checks where applicable.
    Failures go back one round (regenerate question), not silent-dropped.
Phase F  distillation             — unchanged mechanics; RESOLVE FIRST:
    ⚠ generator model = load_checker() at scenario_gen.py:754 (critic-model
    incident). Before any v2 distillation run: patched copy w/ load_teacher()
    or deliberate env override + record generator model id in FROZEN metadata.

## Why the two new gates are non-optional (evidence)

- German-law paper: naive single-prompt synthetic QA made downstream reasoning
  WORSE; only graded + filtered pipelines improved models.
- S³-R1/SAGE: unverified synthetic pairs leak unsolvable/misgrounded items;
  solve-back filtering is what made training data effective (+10–27% deltas).
- CRAG reproduction: LLM relevance judges fail silently on out-of-distribution
  topics → always pair judge output with deterministic hygiene filters.
- Our own history agrees: KC art. 751 sat unused because nobody agentic ever
  asked for limitation periods; KK art. 148a slipped substring gates.

## Build estimate & reuse

- Engine exists and is battle-tested: `agentic_loop.py` + `execute_tool`
  (currently used ONLY by scenario_gen distillation).
- New code = one script per insertion point (curate.py, solve_back.py) in the
  doc-type folder + recipe.md updates. No core changes required.
- Cost note: tool loops multiply API calls — keep iteration caps (max_iter 6),
  parallel workers, and per-doc-type budgets in the manifest.

## Acceptance criteria before v2 adoption

1. Explorer-judge output passes SPOT CHECK B1/B2 (real reading) on a pilot
   doc type with reject-rate and reason-quality comparable to manual curation.
2. Solve-back keeps ≥80% of questions with consistent answers; failures
   inspected by reading, not just counted.
3. Frozen artifacts carry judge/solver metadata (model id, iterations, verdicts).
4. Diary provenance entry per adopted paper (done for #92; keep adding).

---

## UPDATE 2026-08-24 — PHASE BETA design (mutation stage replacement)

User directive: phase2 alternate = find related templates in RAG, then give the
AI instructions AND tools so it grounds first, then emits add/remove/modify ops.

### Retrieval shape ("find all related templates")
- Unbounded "ALL" is not feasible; bounded stack instead:
  1. `get_template(doc_type)` = primary base template (exact lookup)
  2. `semantic_search` top-k≈5 for neighboring/variant templates, deduped
     (near-identical lineage rows collapse)
  3. Clause-level material: the AGENT pulls what it wants via tools
     (this is where agentic freedom lives - not template level)

### Edit taxonomy (validated by literature)
add / remove / modify(replace) as STRUCTURED ops with target location +
new text + cite_refs - CoEdIT (instruction-tuned editing ops), Agent-DocEdit /
DocEdit-v2 (ADD/DELETE/MODIFY/REPLACE command sets executed on grounded regions),
LEDGER ACL-F2026 (⚠ isolated edits break cross-references; dependency-aware
consistency check raised 56->76 consistency). => after edits, run a
cross-reference/coherence pass (§ numbering, defined terms).

### WHAT TO SAVE - answer: BOTH product and trail (research consensus)
1. `extended_template` - synthesized FULL text (coherent whole). Needed by
   phase3 + solve-back gate as conditioning context.
2. `edits.jsonl` sidecar - structured ops {op, target_section, old/new text,
   cite_refs, judge reason} = auditability, spot checks, reproducibility.
3. Provenance-exact chunks - store the CITED CHUNK TEXT ITSELF inside the
   record, not just the ref (CuratorKIT arXiv:2606.21631: storing exact source
   at generation time eliminates the post-hoc-retrieval error class in
   grounding verification; append-only provenance_chain; rejected samples keep
   structured failure reasons - our null-source_ref bug is exactly their
   documented error class).
4. Verification tiers (CORRECTED 2026-08-24 - earlier wording overclaimed):
   FULL deterministic verification of free-text edits is IMPOSSIBLE (models
   adapt wording; verbatim presence/absence checks fail in both directions).
   Tier 1 deterministic = FORMAT/BOOKKEEPING only: ops well-formed, cite_refs
     resolve, cited chunk texts stored in-record, artifact frozen w/ sha.
   Tier 2 probabilistic = NLI / embedding-overlap / judge grounding scores
     with thresholds - CONFIDENCE FILTERS, never proofs (CuratorKIT's
     hallucination gate is itself an LLM judge; LEDGER's consistency needs
     LLM mediation; SAGE/S3-R1 verify by fuzzy solve-and-compare).
   Tier 3 = agent/human READING spot checks as final arbiter (recipe.md).
   Optional diversity trade-off: bias ADDs toward near-verbatim adaptation of
     cited chunks (party/date substitution allowed) -> presence checks become
     ~feasible and legal soundness rises, linguistic diversity falls.
5. Trajectory NOT saved for training here (token-heavy; distillation generates
   its own trajectories later) - only verdicts + final artifacts.

### Papers (this round)
CoEdIT ACL-F2023; LEDGER ACL-F2026 (dependency-aware editing consistency);
DocEdit-v2 EMNLP2024; Agent-DocEdit (OPENREVIEW); ACORD ACL2025 (clause
retrieval for drafting = lawyer workflow; LLM clause-conflict warning);
CuratorKIT arXiv:2606.21631 + arXiv:2606.11127 (provenance-exact gating,
adaptive recovery, manifest+checksums convention - matches our FROZEN style).
