from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from legal_drafter import rag
from legal_drafter.retrieval.hybrid import (
    DEFAULT_COLLECTION,
    QdrantHybridStore,
    build_hybrid_index,
    extract_legal_refs,
    normalize_doc,
    rrf_fuse,
)

RUN_DENSE = os.environ.get("LEGAL_DRAFTER_TEST_DENSE", "0") == "1"


def _write_corpus(lines: list[dict], tmp: Path) -> Path:
    path = tmp / "corpus.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(json.dumps(line, ensure_ascii=False) + "\n")
    return path


class NormalizeTests(unittest.TestCase):
    def test_normalize_maps_source_ref_and_legal_area(self):
        raw = {
            "source_ref": "curated:kc_zlecenie_due_care_750",
            "source_type": "statute_summary",
            "title": "KC art. 734-751",
            "top_category": "service_quality",
            "text": "Przy umowie zlecenia stosuje się reżim starannego działania.",
        }
        doc = normalize_doc(raw)
        self.assertEqual(doc["chunk_id"], "curated:kc_zlecenie_due_care_750")
        self.assertEqual(doc["source_id"], "curated:kc_zlecenie_due_care_750")
        self.assertIn("contracts", doc["legal_area"])
        self.assertEqual(doc["source_type"], "statute_summary")  # original field preserved

    def test_resolve_kc_art_alias(self):
        self.assertEqual(
            rag.resolve_source_id("kc_art_481"),
            "sejm_eli:sejm_eli_du_2024_1061_art_481",
        )


class FilterTests(unittest.TestCase):
    def test_missing_field_does_not_reject(self):
        doc = {"text": "x", "source_type": "statute"}
        self.assertTrue(rag.doc_matches_filter(doc, {"legal_area": ["housing"]}))


class Bm25Tests(unittest.TestCase):
    def test_search_ranks_relevant_doc_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_corpus(
                [
                    {
                        "source_ref": "a",
                        "source_type": "statute",
                        "title": "Kaucja najmu",
                        "text": "Kaucja zabezpiecza należności z tytułu najmu i podlega zwrotowi.",
                    },
                    {
                        "source_ref": "b",
                        "source_type": "statute",
                        "title": "Silnik samochodu",
                        "text": "Silnik spalinowy wymaga wymiany oleju co 15000 km.",
                    },
                ],
                Path(tmp),
            )
            index = rag.build_rag_index([path])
            results = rag.search_rag_index(index, "kaucja najmu zwrot", top_k=2)
            self.assertEqual(rag.source_id(results[0][1]), "a")

    def test_rule_context_resolves_kc_alias_and_includes_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_corpus(
                [
                    {
                        "source_ref": "sejm_eli:sejm_eli_du_2024_1061_art_481",
                        "source_type": "statute",
                        "title": "KC art. 481 odsetki za opóźnienie",
                        "text": "Odsetki za opóźnienie w spełnieniu pieniężnego świadczenia.",
                    },
                    {
                        "source_ref": "z",
                        "source_type": "abusive_clause",
                        "title": "Klauzula",
                        "text": "Inna klauzula niedozwolona.",
                    },
                ],
                Path(tmp),
            )
            index = rag.build_rag_index([path])
            results = rag.search_rule_context(index, "no_contractual_penalty_for_payment_delay", top_k=4)
            ids = [rag.source_id(doc) for _, doc in results]
            self.assertIn("sejm_eli:sejm_eli_du_2024_1061_art_481", ids)


class FusionTests(unittest.TestCase):
    def test_rrf_fuse_rewards_overlap(self):
        fused = rrf_fuse([["x", "y", "z"], ["x", "w"]])
        self.assertGreater(fused["x"], fused["y"])
        self.assertGreater(fused["x"], fused["w"])


class CitationTests(unittest.TestCase):
    def test_extract_legal_refs(self):
        refs = extract_legal_refs("Zgodnie z art. 481 k.c. oraz u.o.p.l. ")
        kinds = {r["kind"] for r in refs}
        self.assertIn("kc", kinds)
        self.assertIn("ustawa_o_ochronie_praw_lokatorow", kinds)


@unittest.skipUnless(RUN_DENSE, "set LEGAL_DRAFTER_TEST_DENSE=1 to run dense retrieval tests")
class DenseTests(unittest.TestCase):
    def test_hybrid_build_and_recall(self):
        with tempfile.TemporaryDirectory() as tmp:
            qdrant_path = Path(tmp) / "qdrant"
            corpus_path = _write_corpus(
                [
                    {
                        "source_ref": f"doc_{i}",
                        "source_type": "statute" if i % 2 == 0 else "abusive_clause",
                        "title": f"Kaucja najmu {i}",
                        "text": "Kaucja zabezpiecza należności z tytułu najmu i podlega zwrotowi w ciągu miesiąca." if i % 3 == 0
                        else "Odsetki za opóźnienie w spełnieniu świadczenia pieniężnego.",
                    }
                    for i in range(24)
                ],
                Path(tmp),
            )
            store = build_hybrid_index([corpus_path], qdrant_path=qdrant_path, rebuild=True)
            self.assertGreater(store.count, 0)
            hits = store.recall("kaucja najmu zwrot", top_k=5)
            self.assertTrue(hits)
            store.close()

    def test_rag_search_uses_dense(self):
        with tempfile.TemporaryDirectory() as tmp:
            qdrant_path = Path(tmp) / "qdrant_b"
            corpus_path = _write_corpus(
                [
                    {
                        "source_ref": f"doc_{i}",
                        "source_type": "statute" if i % 2 == 0 else "abusive_clause",
                        "title": f"Kaucja najmu {i}",
                        "top_category": "deposit_or_advance" if i % 3 == 0 else "payment",
                        "text": "Kaucja zabezpiecza należności z tytułu najmu i podlega zwrotowi w ciągu miesiąca." if i % 3 == 0
                        else "Odsetki za opóźnienie w spełnieniu świadczenia pieniężnego.",
                    }
                    for i in range(24)
                ],
                Path(tmp),
            )
            index = rag.build_rag_index([corpus_path], dense=True, qdrant_path=qdrant_path, rebuild=True)
            self.assertIsNotNone(index["hybrid"])
            results = rag.search_rag_index(index, "kaucja najmu zwrot", top_k=3)
            self.assertTrue(results)
            # legal-area filter still applies through the dense path
            filtered = rag.search_rag_index(index, "kaucja najmu", top_k=3, filters={"legal_area": ["housing"]})
            self.assertTrue(filtered)
            self.assertTrue(all("housing" in (d.get("legal_area") or []) for _, d in filtered))


if __name__ == "__main__":
    unittest.main()
