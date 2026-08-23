from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from .diary import Diary
from .doc_types import DocType, load_doc_types
from .document_specs import load_document_spec
from .io_utils import read_json, read_text, write_json, write_text
from .pipeline import load_prompts
from .renderer import load_locale, render_markdown
from .retrieval.hybrid import (
    DEFAULT_COLLECTION,
    DEFAULT_QDRANT_PATH,
    QdrantHybridStore,
    analyze_corpus,
    build_hybrid_index,
)
from .corpus_ingest import fetch_sejm_act, write_corpus

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def default_corpus_paths() -> list[Path]:
    """All ingested corpus files (recovered main + any acts added via `rag ingest`)."""
    sources_dir = PROJECT_ROOT / "dataset/03_drafting_tasks_sources/sources"
    return sorted(sources_dir.glob("*.jsonl"))
DEFAULT_DIARY = PROJECT_ROOT / ".diary"
DEFAULT_DOC_TYPES = PROJECT_ROOT / "config" / "doc_types" / "public_doc_types.json"
DEFAULT_DOCUMENT_SPECS = PROJECT_ROOT / "config" / "document_specs"
DEFAULT_PROMPTS = PROJECT_ROOT / "config" / "prompts"
DEFAULT_LOCALES = PROJECT_ROOT / "config" / "locales"
DEFAULT_SCHEMAS = PROJECT_ROOT / "config" / "schemas"


def _load_supported_doc(doc_type_id: str, doc_types_path: Path) -> DocType:
    doc_types = load_doc_types(doc_types_path)
    if doc_type_id not in doc_types:
        supported = ", ".join(sorted(doc_types))
        raise ValueError(f"Unsupported doc_type_id={doc_type_id}. Supported: {supported}")
    return doc_types[doc_type_id]


def _sample_clause(section_number: int, section_title: str, topics: list[str], min_clauses: int) -> dict[str, Any]:
    clauses = []
    topic_text = ", ".join(topics[:3]) if topics else section_title.lower()
    for idx in range(1, min(max(min_clauses, 1), 3) + 1):
        clauses.append({
            "id": f"{section_number}.{idx}",
            "text": (
                f"Strony ustalają zasady dotyczące: {topic_text}. "
                "Szczegółowe dane pozostają oznaczone właściwymi placeholderami i wymagają uzupełnienia przed użyciem dokumentu."
            ),
            "source_ids": [],
        })
    return {
        "section_number": section_number,
        "section_title": section_title,
        "clauses": clauses,
    }


def run_smoke_draft(
    *,
    facts_file: Path,
    out_dir: Path,
    doc_type_id: str = "UMOWA_ZLECENIA",
    doc_types_path: Path = DEFAULT_DOC_TYPES,
    document_specs_dir: Path = DEFAULT_DOCUMENT_SPECS,
    prompts_dir: Path = DEFAULT_PROMPTS,
    locales_dir: Path = DEFAULT_LOCALES,
    schemas_dir: Path = DEFAULT_SCHEMAS,
) -> dict[str, Any]:
    doc = _load_supported_doc(doc_type_id, doc_types_path)
    spec = load_document_spec(document_specs_dir / "umowa_zlecenia.v1.json")
    placeholders = read_json(schemas_dir / "placeholders.json")
    facts = read_text(facts_file)
    prompts = load_prompts(prompts_dir)
    locale_pl = load_locale(locales_dir / "pl.json")[doc.id]

    sections: dict[str, dict[str, Any]] = {}
    for idx, section_title in enumerate(doc.sections, start=1):
        section_spec = spec.get("sections", {}).get(str(idx), {})
        if section_spec.get("mode") in {"deterministic_party_block", "deterministic_signature_block"}:
            continue
        sections[str(idx)] = _sample_clause(
            idx,
            section_title,
            list(section_spec.get("topics", [])),
            int(section_spec.get("min_clauses", 1)),
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    draft = render_markdown(
        doc_type_id=doc.id,
        doc_title=locale_pl["title"],
        section_titles=locale_pl["sections"],
        sections=sections,
        party_lines=locale_pl["party_lines"],
        signature_lines=locale_pl["signature_lines"],
        safe_sentences=locale_pl.get("final_safe_sentences", []),
    )
    report = {
        "mode": "smoke",
        "doc_type_id": doc.id,
        "facts_file": str(facts_file),
        "facts_characters": len(facts),
        "prompt_templates_loaded": sorted(prompts.__dict__.keys()),
        "placeholder_count": len(placeholders.get("allowlist", [])),
        "draft_file": "draft_pl.md",
    }
    write_text(out_dir / "draft_pl.md", draft)
    write_json(out_dir / "report.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="legal-drafter")
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("smoke", help="Run a deterministic local smoke draft")
    smoke.add_argument("--facts-file", type=Path, default=PROJECT_ROOT / "examples" / "facts_umowa_zlecenia.txt")
    smoke.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "outputs" / "smoke")
    smoke.add_argument("--doc-type", default="UMOWA_ZLECENIA")

    validate = subparsers.add_parser("validate-config", help="Validate public configuration files can be loaded")
    validate.add_argument("--doc-type", default="UMOWA_ZLECENIA")

    diary = subparsers.add_parser("diary", help="Append-only A/B work diary")
    diary.add_argument("--root", type=Path, default=DEFAULT_DIARY)
    diary_sub = diary.add_subparsers(dest="diary_command", required=True)
    d_append = diary_sub.add_parser("append", help="Append an immutable entry")
    d_append.add_argument("--kind", default="note")
    d_append.add_argument("--text", required=True)
    diary_sub.add_parser("tail", help="Show the last 20 entries")
    diary_sub.add_parser("list", help="List all entry headers")
    diary_sub.add_parser("cat", help="Print the whole diary")
    d_sum = diary_sub.add_parser("summarize", help="Append a summary entry")
    d_sum.add_argument("--text", required=True)
    d_dig = diary_sub.add_parser("compact", help="Append a lossless digest (history kept)")
    d_dig.add_argument("--text", required=True)
    diary_sub.add_parser("stats", help="Show diary statistics")
    d_commit = diary_sub.add_parser("commit", help="Record a commit-awareness snapshot")
    d_commit.add_argument("--note", default="")

    rag = subparsers.add_parser("rag", help="Build/inspect/query the hybrid RAG index (agent-facing, JSON output)")
    rag_sub = rag.add_subparsers(dest="rag_command", required=True)
    rag_ingest = rag_sub.add_parser("ingest", help="Fetch a Sejm act via ELI API and write corpus chunks (JSON)")
    rag_ingest.add_argument("--publisher", default="DU")
    rag_ingest.add_argument("--year", type=int, required=True)
    rag_ingest.add_argument("--position", type=int, required=True)
    rag_ingest.add_argument("--title", required=True, help="Obwieszczenie / act title for the corpus records")
    rag_ingest.add_argument("--display-address", dest="display_address", required=True, help='e.g. "Dz.U. 2024 poz. 1796"')
    rag_ingest.add_argument("--out", type=Path, required=True, help="Output JSONL path")
    rag_build = rag_sub.add_parser("build", help="Embed the corpus into the Qdrant hybrid index")
    rag_build.add_argument("--corpus", nargs="+", type=Path, default=None, help="Corpus JSONL file(s) (default: all known corpora)")
    rag_build.add_argument("--qdrant-path", type=Path, default=DEFAULT_QDRANT_PATH)
    rag_build.add_argument("--collection", default=DEFAULT_COLLECTION)
    rag_build.add_argument("--rebuild", action="store_true", help="Force re-embed even if collection exists")
    rag_check = rag_sub.add_parser("check", help="Profile the corpus (coverage/gaps) as JSON")
    rag_check.add_argument("--corpus", nargs="+", type=Path, default=None, help="Corpus JSONL file(s) (default: all known corpora)")
    rag_stats = rag_sub.add_parser("stats", help="Print corpus + collection stats as JSON")
    rag_stats.add_argument("--qdrant-path", type=Path, default=DEFAULT_QDRANT_PATH)
    rag_stats.add_argument("--collection", default=DEFAULT_COLLECTION)
    rag_search = rag_sub.add_parser("search", help="Query the hybrid index and return ranked hits as JSON")
    rag_search.add_argument("--query", required=True)
    rag_search.add_argument("--corpus", nargs="+", type=Path, default=None, help="Corpus JSONL file(s) (default: all known corpora)")
    rag_search.add_argument("--qdrant-path", type=Path, default=DEFAULT_QDRANT_PATH)
    rag_search.add_argument("--collection", default=DEFAULT_COLLECTION)
    rag_search.add_argument("--top-k", type=int, default=5)
    rag_search.add_argument("--dense", action="store_true", help="Use dense+sparse hybrid recall (else BM25 only)")

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "smoke":
        report = run_smoke_draft(facts_file=args.facts_file, out_dir=args.out_dir, doc_type_id=args.doc_type)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    if args.command == "validate-config":
        doc = _load_supported_doc(args.doc_type, DEFAULT_DOC_TYPES)
        load_document_spec(DEFAULT_DOCUMENT_SPECS / "umowa_zlecenia.v1.json")
        load_prompts(DEFAULT_PROMPTS)
        load_locale(DEFAULT_LOCALES / "pl.json")
        read_json(DEFAULT_SCHEMAS / "placeholders.json")
        print(json.dumps({"status": "ok", "doc_type_id": doc.id}, ensure_ascii=False))
        return 0

    if args.command == "diary":
        d = Diary(args.root)
        if args.diary_command == "append":
            seq = d.append(args.kind, args.text, repo_root=PROJECT_ROOT)
            print(json.dumps({"seq": seq}, ensure_ascii=False))
        elif args.diary_command == "tail":
            for e in d.tail():
                meta = " ".join(f"{k}={v}" for k, v in (e.meta or {}).items())
                print(f"[{e.seq}] {e.ts} ({e.kind}) {meta}\n{e.text}\n")
        elif args.diary_command == "list":
            for e in d.entries():
                first = e.text.splitlines()[0] if e.text else ""
                git = (e.meta or {}).get("git", "")
                print(f"{e.seq}\t{e.kind}\t{(e.meta or {}).get('date','')}\t{git}\t{first}")
        elif args.diary_command == "cat":
            print(d.read_text())
        elif args.diary_command == "summarize":
            seq = d.summarize(args.text)
            print(json.dumps({"seq": seq}, ensure_ascii=False))
        elif args.diary_command == "compact":
            seq = d.compact(args.text)
            print(json.dumps({"seq": seq}, ensure_ascii=False))
        elif args.diary_command == "commit":
            seq = d.commit_snapshot(note=args.note, repo_root=PROJECT_ROOT)
            print(json.dumps({"seq": seq}, ensure_ascii=False))
        elif args.diary_command == "stats":
            print(json.dumps(d.stats(), ensure_ascii=False))
        return 0

    if args.command == "rag":
        try:
            if args.rag_command == "ingest":
                records = fetch_sejm_act(
                    publisher=args.publisher,
                    year=args.year,
                    position=args.position,
                    title=args.title,
                    display_address=args.display_address,
                )
                count = write_corpus(records, args.out)
                print(json.dumps(
                    {
                        "status": "ok",
                        "publisher": args.publisher,
                        "year": args.year,
                        "position": args.position,
                        "records": count,
                        "out": str(args.out),
                    },
                    ensure_ascii=False,
                ))
                return 0
            corpus_paths = args.corpus or default_corpus_paths()
            if args.rag_command == "build":
                store = build_hybrid_index(
                    corpus_paths,
                    qdrant_path=args.qdrant_path,
                    collection_name=args.collection,
                    rebuild=args.rebuild,
                )
                print(json.dumps(
                    {
                        "status": "ok",
                        "collection": store.collection_name,
                        "points": store.count,
                        "corpora": [str(p) for p in corpus_paths],
                    },
                    ensure_ascii=False,
                ))
                return 0
            if args.rag_command == "check":
                report = analyze_corpus(corpus_paths)
                print(json.dumps(report, ensure_ascii=False, indent=2))
                return 0
            if args.rag_command == "stats":
                store = QdrantHybridStore(collection_name=args.collection, path=args.qdrant_path)
                corpus_count = 0
                try:
                    for path in corpus_paths:
                        with open(path, encoding="utf-8") as handle:
                            corpus_count += sum(1 for line in handle if line.strip())
                except FileNotFoundError:
                    corpus_count = None
                print(json.dumps(
                    {
                        "exists": store.exists(),
                        "points": store.count,
                        "corpus_lines": corpus_count,
                        "corpora": [str(p) for p in corpus_paths],
                    },
                    ensure_ascii=False,
                ))
                return 0
            if args.rag_command == "search":
                from . import rag as _rag

                index = _rag.build_rag_index(
                    corpus_paths,
                    dense=args.dense,
                    qdrant_path=args.qdrant_path,
                    collection_name=args.collection,
                )
                results = _rag.search_rag_index(index, args.query, top_k=args.top_k)
                payload = {
                    "ok": True,
                    "query": args.query,
                    "top_k": args.top_k,
                    "dense": index["hybrid"] is not None,
                    "results": [
                        {
                            "rank": i + 1,
                            "source_id": _rag.source_id(doc),
                            "score": round(score, 4),
                            "source_type": doc.get("source_type"),
                            "title": doc.get("title"),
                            "text_preview": _rag.preview(doc.get("text"), 300),
                        }
                        for i, (score, doc) in enumerate(results)
                    ],
                }
                print(json.dumps(payload, ensure_ascii=False, indent=2))
                return 0
        except Exception as exc:  # agent-visible error report
            logging.getLogger("legal_drafter.cli").exception("rag %s failed", args.rag_command)
            print(json.dumps({"ok": False, "error": str(exc), "command": args.rag_command}, ensure_ascii=False))
            return 1

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
