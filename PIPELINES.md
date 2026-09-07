# Pipelines

End-to-end flow (each stage's code, inputs, outputs, and rerun command).
Everything below was verified against code and data on 2026-09-07.
`outputs/` and `scripts/` are gitignored (local experiment state);
the thesis PDF is built from `thesis/` via `./tectonic`.

```
corpus (9 JSONL, 14,197 chunks)
  -> scripts/rag/build_index.py ......... .rag/qdrant (14,197 pts)
  -> scripts/rag/server.py .............. RAG API :10100 (BGE-M3 dense+sparse
                                           + reranker, 27 curated templates)
  -> scripts/question_gen/run_beta_doc.py  1,030 variations (P1 research,
     + seeds (1,459 dirs)                    P2 forced draft, P3 questions)
  -> scripts/question_gen/extract+merge .. questions_merged.jsonl (2,104)
  -> scripts/answer_gen/run_qa_pairs.py .. qa_pairs_full.jsonl (2,148 episodes)
  -> scripts/answer_gen/finalize ......... sft bank (2,111) + holdout 42
  -> scripts/vast/train_lora_vast.py ..... adapters checkpoint-250 + final
  -> scripts/eval/serve_vllm.py .......... student API :18000 (base + LoRAs)
  -> scripts/eval/eval_holdouts_v2_v3.py .. 292 drafts (100/100/92)
  -> scripts/judge_eval100.py + tally .... 23/18/59, 18/16/58, 14/24/54
```

Conventions: long runs go in `tmux` (the shell tool kills timed-out process
trees); never overwrite a run's output dir (new timestamped dir instead);
log runs with `legal-drafter diary append/commit` before and after;
secrets live in `.env` (gitignored — see `.env.example`, incl. the RAG section).

## 1. Corpus

- Sources: `dataset/03_drafting_tasks_sources/sources/` — 9 JSONL files,
  14,197 article-level chunks (base 13,589 + consumer/payment wave +194 +
  min-wage/social-insurance/copyright wave +414). `source_manifest.json`
  is descriptive only; builds glob `*.jsonl`, nothing reads the manifest.
- Ingestion: `src/legal_drafter/corpus_ingest.py` (Sejm ELI two-step:
  `/struct` article walk, then `/text.html` per article) and
  `scripts/fetch_acts_pdf.py` (fallback for textPDF-only acts).

## 2. RAG index + serving

- Library: `src/legal_drafter/retrieval/` — `hybrid.py` (store, builder,
  RRF-fused recall), `embeddings.py` (BGE-M3 dense+sparse encoder +
  bge-reranker-v2-m3; dense+sparse ONLY, no ColBERT channel exists anywhere),
  `agentic_tools.py` (4 tools + OpenAI schemas: keyword_search,
  semantic_search, chunk_read, get_template), `remote.py` (HTTP client),
  `templates.py` (27 hand-curated templates + synthetic `other` fallback).
- Build (GPU box): `PYTHONPATH=src python3 scripts/rag/build_index.py
  --corpus dataset/03_drafting_tasks_sources/sources
  --qdrant-path .rag/qdrant`
  (`--collection/--rebuild/--batch-size/--encode-batch-size/--log-every`).
- Serve: `PYTHONPATH=src python3 scripts/rag/server.py` — env-driven
  (`RAG_REPO/CORPUS_DIR/QDRANT_PATH/COLLECTION/HOST/PORT`, defaults:
  repo-root paths, `0.0.0.0:10100`). WARNING: it takes no `--help` and
  starts serving (building the index first if missing) on launch.
  Routes: `keyword_search`, `semantic_search`, `chunk_read`,
  `get_template` (curated 27 only), `/health` (reports point count).
- Live index: `.rag/qdrant` holds the healthy 14,197-point collection
  (restored 2026-09-07 from `.rag/qdrant_backup.tar.gz` after local copies
  went corrupt; the pre-existing state is preserved under
  `outputs/legacy/qdrant_corrupt_partial_20260907/`).
- Clients point at it via `RETRIEVAL_ENDPOINT` (generation pipelines) or
  `RAG_URL` (eval); see the RAG section of `.env.example`.

## 3. Questions (scenario directions -> layperson questions)

- Runner: `scripts/question_gen/run_beta_doc.py` — env-driven, no CLI args:
  `DOC_TYPE`, `SCENARIOS` (directions file) or `hints.txt`, `OUT_DIR`
  (default `outputs/anchors/<doc_type>`), `BETA_RUNS`, `BETA_WORKERS`,
  teacher via `CHECKER_MODEL` + `TEACHER_BASE_URL` (+ keys),
  `RETRIEVAL_ENDPOINT`. Per direction: P1 research loop (cap 5 successful
  calls, call cache, read-set, 3-empty skip) → P2 forced no-tools JSON draft
  (`summary` + `contract`) → P3 backtranslated `prosta`/`szczegółowa`
  questions. Writes `variation_N.json` + `trajectory_N.json` per direction.
- Seeds (hand-written, no generator in repo): 25×
  `outputs/anchors/*/scenarios.txt` (1,459 lines) +
  `outputs/anchors/umowa_zlecenia/beta_experiment/hints.txt` (55);
  skips in per-type `not_applicable.txt` (61).
- Glue: `scripts/question_gen/extract_questions.py --anchors outputs/anchors
  --out questions_all.jsonl` (flattens variations to batch rows) and
  `scripts/question_gen/merge_questions.py --in a.jsonl b.jsonl
  --out questions_merged.jsonl --manifest manifest.json` (concatenates,
  stamps per-file 1-based `id` + `source_file`).
- Canonical output: `outputs/merged_question_set/questions_merged.jsonl`
  (2,104 = 864 + 248 + 398 + 550 + 44 retry; batch inputs preserved under
  `outputs/legacy/question_set/`). Re-running extract+merge reproduces it
  content-exactly (verified).

## 4. Answers -> frozen bank

- Runner: `scripts/answer_gen/run_qa_pairs.py` — env: `QA_QUESTIONS`
  (default `outputs/merged_question_set/questions_merged.jsonl`),
  `QA_OUT_DIR`, `QA_WORKERS`, `QA_TOOL_CAP`, teacher env,
  `RETRIEVAL_ENDPOINT`. Imports 11 helpers from `run_beta_doc` (resolved
  via `scripts/question_gen` on its path). Each question gets one agentic
  answer episode (Opus 4.8 historically). Raw output:
  `outputs/qa_pairs_all/qa_pairs_full.jsonl` (2,148 rows).
- Freeze: `scripts/answer_gen/finalize_qa_pairs_all.py` (no args; drops
  rows without message payloads = transport errors, caps tool observations
  at 2000 chars for the train view). Output:
  `outputs/qa_pairs_all_final/` — `sft_final.jsonl` (full) +
  `sft_train_view.jsonl` (capped) + `manifest.json` (2,111 rows).
- Holdout: seed-3407 shuffle of the bank → 42 ids in
  `outputs/eval100/holdout_ids_true42.json` (reproducible one-liner;
  the stale `holdout_ids.json` from an older split is quarantined under
  `outputs/legacy/` — never use it).

## 5. Training

- `scripts/vast/train_lora_vast.py` on 4× RTX PRO 5000 (DDP): Unsloth
  Qwen3.5 recipe (r16/α16/dropout 0/all-linear/bias none, adamw_8bit,
  lr 2e-4, cosine, warmup 25, batch 1×grad-accum 2, 2 epochs, seq 32768,
  answer-only loss incl. reasoning, seed 3407). Args: `--data/--output/
  --model/--max-seq/--epochs/--lr/--lora-r/--save-steps/--seed/
  --batch-size/--grad-accum`. Saves `checkpoint-{N}` + `final`
  (~258 steps/epoch, ~516 total; evaluated: checkpoint-250, final).
- Weights home: `models/` (gitignored; place `checkpoint-250/` + `final/`
  + trainer logs here — currently awaiting upload).

## 6. Model serving (eval)

- `scripts/eval/serve_vllm.py` (env `VLLM_*`, `--dry-run` to preview):
  serves base + LoRA modules in one engine — port 18000, ctx 32768,
  rank 16, `--enable-auto-tool-choice --tool-call-parser qwen3_coder`
  (Qwen3.5 needs the coder parser; hermes fails silently),
  `--gpu-memory-utilization 0.85` (1.0 OOMs mid-eval). Served names must
  match the eval `--models` list (base path + `ckpt250`/`final`).

## 7. Eval (answering eval100)

- Runner: `scripts/eval/eval_holdouts_v2_v3.py` — sends system + question
  only; research loop (cap 5, ≤10 turns) + separate forced no-tools draft;
  needs the vLLM + RAG servers live (SSH tunnels work).
  Args: `--n/--parallel/--models/--input/--out` + sampling
  (`--research-temp/--draft-temp` default 0.3/0.7,
  `--research-max-tokens/--draft-max-tokens` default 32768,
  `--top-p/--top-k/--min-p/--presence/--repetition` = Unsloth Qwen3.5
  thinking-mode recommendations). Every row records its effective spec.
- Inputs: `outputs/eval100/eval100_questions.jsonl` (42 holdout + 58
  external; external built by `scripts/build_external_58.py`).
- Outputs: `outputs/eval100/eval100_{base,lora250,final}.jsonl`
  (100/100/92 rows; 8 final-adapter rows are vLLM server disconnects).

## 8. Judge (pairwise comparison + tally)

- `scripts/judge_eval100.py prepare|run` (`JUDGE_*` env or flags):
  `prepare` joins questions + per-model contracts into `judge_inputs.jsonl`;
  `run` judges every set with 2 judges × 2 orders × 2 trials (8 verdicts),
  resume-safe, `--retry-errors` replays failed cells. Judges: GPT 5.5
  (`judge_a`) + GLM 5.1 (`judge_b`) via provider router, temp 0.0.
- `scripts/tally_judge.py --annotations ... --out ...`: strict rule —
  a win needs both judges + both orders agreeing with unanimous cells,
  else a flagged tie. Frozen run:
  `outputs/judge_runs/eval100_clean_w3/annotations_clean.jsonl`
  (2,272 verdicts → 23/18/59, 18/16/58, 14/24/54; 171 ties).
- `scripts/make_blind_ab.py`: renders blind A/B packs for manual reads.

## Retired (do not use)

- `outputs/legacy/modal_superseded/` — all Modal jobs (build/serve/train/
  infer/download). Nothing live imports them.
- `outputs/legacy/template_search_ccby_SUPERSEDED.py` + CC-BY template
  collection — superseded by the 27 curated templates.
- `scripts/vast/eval_holdouts{,_v2*}.py`, `src/.../scenario_gen.py`,
  `agentic_loop.py`, phase scripts, `run_qa_pairs_pl*.py`,
  `anchor_pool*` files — superseded experiment stages.
