# Polish Legal Document Drafting Assistant

Research prototype for structured Polish legal-document drafting with
deterministic rendering, retrieval-aware validation, and verifier scaffolding.
Developed as part of an engineering thesis on synthesising grounded,
tool-trajectory training data for a small agentic drafting model.

This repository ships the code path **plus the frozen research artifacts**
(training bank, evaluation sets, corpus sources, anchors, LoRA adapters) so
that every number in the thesis is reproducible from the repo contents.

This is a research prototype, not legal advice.

## What is here

```text
src/legal_drafter/          Python package: pipeline, rendering, retrieval, validation, verifier prototypes
config/                     Document-type registry, drafting specs, prompts, JSON schemas, PL/EN locales
scripts/                    Full pipeline code (all stages below)
dataset/03_drafting_tasks_sources/
                            Frozen RAG corpus: 9 JSONL files, 14,197 chunks
                            (statute articles ~7,500, UOKiK abusive clauses >6,600, judgments, forms)
outputs/qa_pairs_all_final/
                            Frozen SFT bank: 2,111 training episodes / 16,308 tool calls,
                            25 document types (gzipped + SHA-256 sidecars)
outputs/eval100/            Frozen evaluation: 100 questions + base/lora250/final drafts,
                            external 58-question set, 42 holdout IDs (with sidecars)
outputs/anchors/            1,459 hand-written drafting directions, 1,030 completed
                            trajectory/draft/question-pair variations, frozen anchor sets
models/                     Trained LoRA adapters (checkpoint-250, final) for Qwen3.5-2B
thesis/                     LaTeX sources + PDF (local; not tracked)
```

Key docs:

- `PIPELINES.md` — end-to-end stage-by-stage flow with rerun commands.
- `EVAL_RECIPE.md` — how to serve base/LoRA checkpoints with vLLM and replay eval100.
- `docs/ARCHITECTURE.md`, `docs/VERIFIER_SPEC.md` — design notes.

## Pipeline (summary)

```text
corpus (9 JSONL, 14,197 chunks)
  -> scripts/rag/build_index.py            BGE-M3 dense+sparse index + reranker, 27 curated templates
  -> scripts/question_gen/                 1,459 human directions -> 1,030 agentic variations
                                           (P1 retrieval research -> P2 forced draft -> P3 question backtranslation)
  -> scripts/answer_gen/                   2,111-episode SFT bank (tool trajectory + grounded draft + question pair)
  -> scripts/vast/train_lora_vast.py       LoRA fine-tune on 2,069 train episodes (42 held out)
  -> scripts/eval/                         100-question replay under three served models
  -> scripts/judge_eval100.py + tally_judge.py
                                           blinded two-judge comparison (replayed + external set)
```

## Results (thesis headline)

Three-way blinded judge comparison on the 100-question replay: decisive win
rates base-vs-lora250 0.561, base-vs-final 0.529, lora250-vs-final 0.368 —
all within the ±0.22–0.24 detectable-effect band, i.e. no configuration is
distinguishable at this sample size. Full analysis and error accounting are in
`thesis/WUT-Thesis/` (built locally with `./tectonic`).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
cp .env.example .env
```

## Run the public smoke path

```bash
legal-drafter validate-config
legal-drafter smoke --facts-file examples/facts_umowa_zlecenia.txt --out-dir outputs/smoke
```

The smoke path supports `UMOWA_ZLECENIA` (mandate/work-services agreement) only
and does not call an LLM: it verifies that public configuration loads correctly
and writes a deterministic sample draft/report under `outputs/`, which is
ignored by Git. The 25 supported document types of the full system live in the
shipped specs/templates and the frozen data.

## Verify shipped artifacts

Every frozen dataset has a SHA-256 sidecar next to it:

```bash
cd outputs/qa_pairs_all_final && sha256sum -c *.sha256
cd outputs/eval100 && sha256sum -c *.sha256
# gzipped banks: gunzip -k sft_train_view.jsonl.gz first
```

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

## Safety and limitations

- Generated documents are drafts for research/evaluation only.
- A qualified lawyer must review any document before real use.
- Retrieval and verifier components are prototypes and require broader evaluation.
- Not tracked: `.env`/API tokens, raw generation dumps, training logs, local
  diaries, and the `thesis/` build products.
