# eval100 — clean external benchmark set

The 100-question base-vs-LoRA benchmark lives here. Everything in
`outputs/legacy/contaminated_holdout_evals_20260903/` is quarantined — do not
use it for quality comparisons.

## Why this exists

The prior eval accidentally measured LoRA on trained-on questions (see
`outputs/legacy/contaminated_holdout_evals_20260903/README.md`). The training
run's own holdout was discovered on the training box and is the only genuinely
unseen data from the original pool.

## Files

### `holdout_ids_true42.json`

The **42 genuinely unseen question ids** from the ckpt250 training run.

- Provenance: downloaded from the vast.ai training box
  (`/workspace/checkpoints-2b/holdout_ids.json`, instance 49589642,
  2026-09-03) and independently verified: re-running
  `random.Random(3407).shuffle` over `outputs/qa_pairs_all_final/sft_train_view.jsonl`
  (2111 lines) reproduces the exact same 42 ids.
- **These are id *values*, not line indices.** Any eval script must match
  them against the row `id` field — `eval_holdouts_v2_v3.py`'s
  `lines[i]` indexing is exactly the bug that caused the contamination.
- From the 2,104-row pool, these 42 were excluded from training
  (`train=2069 holdout=42` in `outputs/train_5090_run.log`).

### (pending) `question_ids.json`

The full 100-question set: these 42 + 58 newly written external questions
(4 per doc_type × 25 types, seeded sampling, `source: external_eval_v1`).
Status: to be generated next.

## Rules for anything added here

1. Every question must be verifiably absent from
   `outputs/qa_pairs_all_final/sft_train_view.jsonl` (id match + text check).
2. Ids from the original pool must come from `holdout_ids_true42.json` only.
3. New questions get a separate id namespace (`ext_*`) and
   `source: external_eval_v1`.
