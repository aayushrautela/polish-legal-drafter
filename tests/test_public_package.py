import tempfile
import unittest
from pathlib import Path

from legal_drafter.cli import DEFAULT_DOC_TYPES, run_smoke_draft
from legal_drafter.doc_types import load_doc_types
from legal_drafter.legal_warnings import validate_legal_warnings, warning_blockers


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class PublicPackageTests(unittest.TestCase):
    def test_public_doc_types_only_include_supported_umowa_zlecenia(self):
        doc_types = load_doc_types(DEFAULT_DOC_TYPES)
        self.assertEqual(sorted(doc_types), ["UMOWA_ZLECENIA"])
        self.assertEqual(doc_types["UMOWA_ZLECENIA"].title, "Umowa zlecenia")

    def test_smoke_draft_writes_markdown_and_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            report = run_smoke_draft(
                facts_file=PROJECT_ROOT / "examples" / "facts_umowa_zlecenia.txt",
                out_dir=tmp_path,
            )

            draft = (tmp_path / "draft_pl.md").read_text(encoding="utf-8")
            self.assertEqual(report["mode"], "smoke")
            self.assertEqual(report["doc_type_id"], "UMOWA_ZLECENIA")
            self.assertIn("# Umowa zlecenia", draft)
            self.assertIn("[[PARTY_A_NAME]]", draft)
            self.assertTrue((tmp_path / "report.json").exists())

    def test_legal_warning_blocker_for_monetary_contractual_penalty(self):
        sections = {
            "5": {
                "clauses": [
                    {
                        "id": "5.1",
                        "text": "Kara umowna przysługuje za opóźnienie w płatności wynagrodzenia.",
                        "source_ids": [],
                    }
                ]
            }
        }

        warnings = validate_legal_warnings(sections)
        blockers = warning_blockers(warnings)
        self.assertTrue(any(item["type"] == "monetary_contractual_penalty" for item in blockers))


if __name__ == "__main__":
    unittest.main()
