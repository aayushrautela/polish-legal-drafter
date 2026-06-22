# Polish Legal Document Drafting Assistant

Prototype thesis project for generating structured Polish legal-document drafts with retrieval-assisted checks and deterministic rendering.

This repository contains the public, reproducible code path only. Large datasets, model checkpoints, raw experiment outputs, and local notes are intentionally excluded from Git tracking.

## Scope

The project explores a human-in-the-loop legal drafting pipeline:

```text
user facts
  -> document type + requirements
  -> optional legal-source retrieval
  -> structured JSON drafting
  -> validation and warning checks
  -> deterministic Markdown rendering
```

Current focus:

- Polish contract/document drafting experiments.
- JSON-first generation instead of free-form document text.
- Deterministic rendering of headings, party blocks, signatures, and safety text.
- Local retrieval and verifier prototypes for legal-source grounding.

This is a research prototype, not legal advice.

## Repository layout

```text
src/              Core pipeline, rendering, retrieval, validation, verifier prototypes
prompts/          Prompt templates used by the drafting pipeline
requirements/     Document-specific drafting requirements
schemas/          JSON schemas and legal-object taxonomies
locales/          Polish/English rendering labels
examples/         Small sanitized input examples
ARCHITECTURE.md   System architecture and design notes
VERIFIER_SPEC.md  Verifier design notes
```

## What is intentionally excluded

The following are kept local and should not be pushed to GitHub:

- `.env` and API tokens.
- `outputs/`, `outputs_backup/`, and raw generation dumps.
- Large datasets and model checkpoints.
- Local diaries/status notes.
- Python caches and temporary files.

Use `.env.example` as the template for local configuration.

## Basic setup

```bash
python -m venv .venv
source .venv/bin/activate
cp .env.example .env
```

Runtime model-serving code and large experiment assets are kept outside this public branch until they are cleaned into a reproducible release.

Generated outputs should go under `outputs/`, which is ignored by Git.

## Safety and limitations

- Generated documents are drafts for research/evaluation only.
- A qualified lawyer must review any document before real use.
- Retrieval and verifier components are prototypes and require broader evaluation before relying on them.
- Public GitHub history should contain only code, small sanitized examples, and concise documentation.
