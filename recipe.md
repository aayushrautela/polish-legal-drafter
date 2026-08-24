# Recipe: per-doc-type question bank + distillation

Working recipe for producing SFT/distillation data one Polish legal document
type at a time (umowa_o_dzielo done, umowa_zlecenia in progress). Written for
agents following in later sessions. Reflects the **final corrected** pipeline
as of 2026-08-24 — including every bug we hit and fixed. Do not re-derive;
follow this.

## Ground rules (non-negotiable)

- **RAG runs on Modal only.** Never load BGE-M3/reranker locally (~3.2 GB box OOMs).
  Bulk sweeps use the GPU service (`…retrievalservicegpu-app.modal.run`, L4/L40S);
  production serving uses the CPU URL. Both come from `modal deploy modal_retrieval.py`.
- **tmux detached launches only** (opencode kills process groups otherwise).
  Never poll — read persisted logs when asked for status.
- **Never overwrite anything**, especially other agents' outputs. Every round gets
  NEW filenames; superseded artifacts go to `v1_archive/`. Frozen files are immutable.
- **Spot check means actually reading** the content (full text/snippets, judging
  relevance in-context), not running regex/counters. Regex gates are necessary,
  never sufficient. Quote excerpts in the diary as evidence of reading.
- **One phase at a time.** Verify phase N output before launching N+1.
- Log milestones: `PYTHONPATH=src .venv/bin/python -m legal_drafter.cli diary append …`
  (never `python -m legal_drafter.diary` — silent no-op).

---

## Phase A — corpus growth (only when coverage gaps are proven)

1. Web-verify exact Dz.U. positions FIRST (ISAP/eli.gov.pl/dziennikustaw.gov.pl),
   confirm license situation (statutes = public domain, pr. aut. art. 4 ust. 2).
2. Fetch into `dataset/03_drafting_tasks_sources/sources/*.jsonl`:
   - preferred: `legal-drafter rag ingest --year Y --position P …` (ELI `/struct` route)
   - fallback when `/struct` is 404 (textPDF-only acts): `fetch_acts_pdf.py`
     pattern (pypdf extract → split on `Art\. \d+[a-z]*`, same record schema,
     `origin="sejm_eli_pdf"`). One-off regulations may need single-chunk handling (`§` text).
3. Clean before indexing: dedupe duplicate chunk_ids keeping the LONGEST text;
   drop tiny entry-into-force fragments (<~260 chars containing "wchodzi w życie").
   Update `source_manifest.json`.
4. Full index rebuild on Modal GPU (incremental upsert is FORBIDDEN — the indexer
   uses sequential int point ids, an upsert would overwrite existing points):
   `modal run --detach modal_build_index.py` — **--detach is mandatory**
   (plain `modal run` cancels the spawned call when its local entrypoint exits;
   cost us two dead runs).
   The script now replaces ONLY `collection/legal_corpus` and preserves sibling
   collections (templates-wipe incident patched 2026-08-23) — still verify.

**SPOT CHECK A (read):** tail `modal volume get legal-drafter-rag /build.log -`
(points count ≈ old + added); then run 2–3 semantic queries against the endpoint
about the newly ingested topics and READ the top hits — new acts must dominate;
call `get_template(<doc_type>)` and read that templates collection survived.

## Phase B — anchor candidate sweep (retrieval-only, NO LLM)

1. Adapt `outputs/anchors/umowa_zlecenia/build_zlecenie_anchors.py` into the new
   doc-type folder: ~30 kind-balanced queries; strict topical gate with whole-act
   allowlist for acts ingested specifically for this type; per-article cap 4;
   target 60–75.
   Classifier order matters (bugs we fixed): special domain kinds (tax/taxzus)
   with INFLECTION-TOLERANT term lists → statute → generic keywords → clauses.
   Empty kind buckets must be guarded in the round-robin enrichment (ZeroDivision).
   `source_ref` must fall back to `chunk_id` everywhere (null refs poisoned
   citations once — 392 of them).
2. Launch via tmux with `RETRIEVAL_ENDPOINT=<GPU url>`.

**SPOT CHECK B1 (read ALL picks):** dump every picked anchor (id, kind, ~150 chars)
and genuinely read the list. Reject families known to recur: wrong-domain clauses
wearing the right party label (estate brokerage/courier/insurance/TOS "Zleceniodawca"),
statutes outside the relevant range (agencyjna/komis chapters for zlecenie),
criminal-code substring traps (KK art. 148a matched "zleceni" via *zlecenie
zabójstwa*), near-duplicates. Expect to reject 20–50%.

**SPOT CHECK B2 (local audit before freezing):** grep ALL `dataset/**/sources/*.jsonl`
for the doc-type terms and compare against picked ids. This catches what retrieval
misses: id-less rows (`cleaned_rag_chunks.jsonl` has many) and inflection variants.
Hand-read what surfaces; admit genuinely relevant finds (this found KC art. 751,
2-year limitation — core rule no query had surfaced). Also read full text of every
admit from Phase A acts (we caught a whole PIT article family that was actually
about asset amortization, not taxation).

## Phase C — freeze

Write `anchors.final_vXNN.jsonl` + `FROZEN(_VX).json` (sha256, n_anchors, utc) +
a dated section in `merge_manifest.json` listing EVERY admit and drop WITH reasons.
Backfill `source_ref := chunk_id` where null. Frozen files are never edited again.

## Phase D — phase2 mutations (LLM, forced tool)

Run the **copy** inside the doc-type folder:

```
phase2_batch.py --in <frozen anchors> --out <type>_vX.phase2.jsonl \
    --batch 1 --related 8 --parallel 6 --max-tokens 12000
```

- `--batch 1` is a USER DIRECTIVE: ONE anchor per completion. Packing several
  anchors into one completion caused cross-anchor semantic mixing that no
  citation gate can detect. `--parallel 6` = six API calls in flight (that was
  the original intent of "parallelize 5").
- `--related 8`: show the FULL clause bank so the model can ground adds in real
  retrieved material (bank reuse went 7 → 92 with this).
- The copy contains `_valid_add` hard-gating empty/unresolvable `cite_ref` and
  null-safe label resolution. Keep those.

**SPOT CHECK D (counts + reading):** grounding stats over ALL adds must show
0 empty citations; then READ ~10 add/removes across kinds (include the newly
ingested dimensions) and confirm each condition actually matches its cited
clause's content. Baseline to beat: odzielo had 119 adds / 0 empty / 80 bank-cites.

## Phase E — phase3 questions (LLM, forced tool)

```
phase3_instruct.py --in <phase2 out> --out <type>_vX.phase3.jsonl \
    --batch 1 --per-anchor 2 --parallel 6 --max-tokens 12000
```

Automated gates that MUST pass (necessary, not sufficient): strict citation
regex (`art\.?\s*\d+|Dz\.U\.|ustawa z dnia|kodeks`) = 0 hits; length window
(~40–700 chars); 80-char-prefix uniqueness.

**SPOT CHECK E (read 10–15 questions):** layman voice (no lawyer phrasing like
"strona ma obowiązek…"), no citations, each question traceable to ITS OWN
anchor's mutations (no cross-anchor bleed), both variants distinct in axis not
paraphrase. Read ≥3 questions per newly ingested legal dimension.

## Phase F — distillation (the agentic stage — THIS is where tool-using AI lives)

`scenario_gen.py` + `agentic_loop.run_agentic()`: per question, the model loops
(max 6 iterations) calling native tools — `get_template` (doc_type auto-injected),
`semantic_search`, `keyword_search`, `chunk_read` (read-set tracker) — then a
hard final-step forcing function demands the complete contract JSON. Full native
transcript is kept as the training record.

⚠️ **MODEL WIRING TRAP:** `scenario_gen.py:754` builds the generator config via
`load_checker()` — agent 1's odzielo_distill_v1 (92 pairs) was therefore produced
by the CRITIC model, unintentionally. For future runs either use a patched COPY
with `load_teacher()` or set the env deliberately, and RECORD which model
generated each dataset file in the diary + FROZEN metadata.

Command shape:

```
scenario_gen --instructions <phase3.jsonl> --n <len> --parallel 5 \
    --max-tokens 16384 --retrieval-url <GPU endpoint> \
    --out outputs/scenarios/<type>_distill_v1.jsonl --log <same>.log
```

Resume-safe by default (skip completed task_ids); use `--fresh` only intentionally.

**SPOT CHECK F (read 3–5 records END TO END):** trajectory sane (tool calls
sensible, ≤6 iterations, sources resolve), `expected_output` is a COMPLETE
contract (all standard sections, CC-BY-4.0 attribution preserved, template
adapted not pasted), facts match the instruction. Freeze with a FROZEN.json
(sha256 + record count + generator model id).

---

## Current state (2026-08-24)

| artifact | status |
|---|---|
| o_dzielo | 46 anchors → 92 questions → 92 distilled pairs (checker model — known caveat) |
| zlecenia corpus | min-wage act, ZUS act, copyright act, 2026 rates ingested; legal_corpus=14197 pts |
| zlecenia anchors | `anchors.final_v266.jsonl` FROZEN (65; taxzus 12) sha `a91a6f58…`; builder verified null-free by clean rerun |
| zlecenia phase2 | batch-1 rerun IN FLIGHT (grounded variant archived in `v1_archive/batched/`) |
| zlecenia phase3 | waiting for phase2 verification |
| zlecenia distill | blocked on teacher-vs-checker decision |
