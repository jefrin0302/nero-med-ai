"""
test_phase6_audit_edge_cases.py

Focused Regression Test Suite for Phase 6 Step 2.2:
Edge cases, citation structure variations, and evidence integrity.

Tests:
1. Multiple claims in one bullet point separated by semicolon.
2. Citations on separate lines from claims.
3. Multiple citations where one citation has an invalid page bound.
4. Numerical ranges and units matching and mismatching.
5. A footer with multiple source references (excluded as non-clinical).
6. Missing full passage text (missing evidence never becomes positive support).
7. Truncated passage text explicitly marked as truncated (insufficient evidence flagged).
"""

import os
import sys
import unittest

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from clinical_benchmarks.evaluator import ClinicalBenchmarkEvaluator


class TestPhase6AuditEdgeCases(unittest.TestCase):
    def setUp(self):
        self.evaluator = ClinicalBenchmarkEvaluator(rag_system=None)

    def test_01_multiple_claims_in_one_bullet(self):
        """Verifies that multiple claims separated by semicolons in a single bullet are split and evaluated."""
        case = {
            "case_id": "MOCK-EDGE-01",
            "domain": "DOM-01",
            "is_mock": True,
            "is_in_corpus": True,
            "clinician_evaluation": {"review_status": "draft_pending_clinician_review", "is_clinician_approved": False}
        }
        reply = (
            "• Kayser-Fleischer rings reflect copper accumulation in the cornea [AASLD WD Guidelines.pdf, p. 5]; "
            "D-penicillamine initial dose is 250-500 mg daily [EASL-ERN Clinical Practice Guidelines on Wilson's disease.pdf, p. 12]."
        )
        passages = [
            "[Source: AASLD WD Guidelines.pdf | Page: 5 ] Kayser-Fleischer rings reflect copper accumulation in the cornea.",
            "[Source: EASL-ERN Clinical Practice Guidelines on Wilson's disease.pdf | Page: 12 ] D-penicillamine initial dose is 250-500 mg daily."
        ]
        tel = {"retrieved_passages": passages, "evidence_coverage": {}}

        res = self.evaluator.evaluate_case(case, rag_reply=reply, telemetry=tel)

        det = res["details"]["claim_level_support_details"]
        self.assertEqual(det["total_cited_sentences"], 2)
        self.assertEqual(det["supported_count"], 2)
        self.assertEqual(res["metrics"]["claim_level_citation_support"], 1.0)

    def test_02_citations_on_separate_lines_from_claims(self):
        """Verifies that citations on separate lines immediately after claims are correctly associated."""
        case = {
            "case_id": "MOCK-EDGE-02",
            "domain": "DOM-01",
            "is_mock": True,
            "is_in_corpus": True,
            "clinician_evaluation": {"review_status": "draft_pending_clinician_review", "is_clinician_approved": False}
        }
        reply = (
            "• Kayser-Fleischer rings reflect copper accumulation in the cornea.\n"
            "[AASLD WD Guidelines.pdf, p. 5]"
        )
        passages = ["[Source: AASLD WD Guidelines.pdf | Page: 5 ] Kayser-Fleischer rings reflect copper accumulation in the cornea."]
        tel = {"retrieved_passages": passages, "evidence_coverage": {}}

        res = self.evaluator.evaluate_case(case, rag_reply=reply, telemetry=tel)

        det = res["details"]["claim_level_support_details"]
        self.assertEqual(det["total_cited_sentences"], 1)
        self.assertEqual(det["supported_count"], 1)
        self.assertEqual(res["metrics"]["claim_level_citation_support"], 1.0)

    def test_03_multiple_citations_where_one_has_invalid_page(self):
        """Verifies that if one citation in a multi-citation sentence has an invalid page, support fails."""
        case = {
            "case_id": "MOCK-EDGE-03",
            "domain": "DOM-06",
            "is_mock": True,
            "is_in_corpus": True,
            "clinician_evaluation": {"review_status": "draft_pending_clinician_review", "is_clinician_approved": False}
        }
        # Page 12 is valid, but page 99 exceeds AASLD max page 23
        reply = "D-penicillamine promotes urinary copper excretion [AASLD WD Guidelines.pdf, p. 12] [AASLD WD Guidelines.pdf, p. 99]."
        passages = ["[Source: AASLD WD Guidelines.pdf | Page: 12 ] D-penicillamine promotes urinary copper excretion."]
        tel = {"retrieved_passages": passages, "evidence_coverage": {}}

        res = self.evaluator.evaluate_case(case, rag_reply=reply, telemetry=tel)

        det = res["details"]["claim_level_support_details"]
        self.assertEqual(det["total_cited_sentences"], 1)
        self.assertEqual(det["supported_count"], 0)
        self.assertEqual(res["metrics"]["claim_level_citation_support"], 0.0)
        self.assertIn("page_out_of_bounds", det["details"][0]["reason"])

    def test_04_numerical_ranges_and_units(self):
        """Verifies validation of numerical ranges and units."""
        case = {
            "case_id": "MOCK-EDGE-04",
            "domain": "DOM-06",
            "is_mock": True,
            "is_in_corpus": True,
            "clinician_evaluation": {"review_status": "draft_pending_clinician_review", "is_clinician_approved": False}
        }
        # Case A: Matching range 250-500 mg
        reply_good = "D-penicillamine initial dose is 250-500 mg daily [AASLD WD Guidelines.pdf, p. 12]."
        passages = ["[Source: AASLD WD Guidelines.pdf | Page: 12 ] D-penicillamine initial dose is 250-500 mg daily."]
        tel = {"retrieved_passages": passages, "evidence_coverage": {}}

        res_good = self.evaluator.evaluate_case(case, rag_reply=reply_good, telemetry=tel)
        self.assertEqual(res_good["metrics"]["claim_level_citation_support"], 1.0)

        # Case B: Fabricated range 1000-1500 mg
        reply_bad = "D-penicillamine initial dose is 1000-1500 mg daily [AASLD WD Guidelines.pdf, p. 12]."
        res_bad = self.evaluator.evaluate_case(case, rag_reply=reply_bad, telemetry=tel)
        self.assertEqual(res_bad["metrics"]["claim_level_citation_support"], 0.0)
        self.assertIn("unsupported_numerical_threshold", res_bad["details"]["claim_level_support_details"]["details"][0]["reason"])

    def test_05_footer_with_multiple_source_references(self):
        """Verifies that a bibliographic footer with multiple citations is excluded from clinical claims."""
        case = {
            "case_id": "MOCK-EDGE-05",
            "domain": "DOM-01",
            "is_mock": True,
            "is_in_corpus": True,
            "clinician_evaluation": {"review_status": "draft_pending_clinician_review", "is_clinician_approved": False}
        }
        reply = "Sources: AASLD WD Guidelines.pdf, p. 12; EASL-ERN Clinical Practice Guidelines on Wilson's disease.pdf, p. 3"
        passages = ["[Source: AASLD WD Guidelines.pdf | Page: 12 ] Some text."]
        tel = {"retrieved_passages": passages, "evidence_coverage": {}}

        res = self.evaluator.evaluate_case(case, rag_reply=reply, telemetry=tel)

        det = res["details"]["claim_level_support_details"]
        # Bibliographic footer is excluded, so 0 clinical cited sentences were extracted
        self.assertEqual(det["total_cited_sentences"], 0)
        self.assertIsNone(det["score"])

    def test_06_missing_full_passage_text(self):
        """Verifies that missing passage text never becomes a positive support result."""
        case = {
            "case_id": "MOCK-EDGE-06",
            "domain": "DOM-01",
            "is_mock": True,
            "is_in_corpus": True,
            "clinician_evaluation": {"review_status": "draft_pending_clinician_review", "is_clinician_approved": False}
        }
        reply = "Kayser-Fleischer rings indicate copper deposition [AASLD WD Guidelines.pdf, p. 5]."
        # Passage text is completely missing from retrieved passages
        tel = {"retrieved_passages": [], "evidence_coverage": {}}

        res = self.evaluator.evaluate_case(case, rag_reply=reply, telemetry=tel)

        det = res["details"]["claim_level_support_details"]
        self.assertEqual(det["supported_count"], 0)
        self.assertEqual(res["metrics"]["claim_level_citation_support"], 0.0)
        self.assertIn("unretrieved_passage", det["details"][0]["reason"])

    def test_07_truncated_passage_explicitly_marked_as_truncated(self):
        """Verifies that a truncated passage with missing claim terms fails support and flags truncation."""
        case = {
            "case_id": "MOCK-EDGE-07",
            "domain": "DOM-01",
            "is_mock": True,
            "is_in_corpus": True,
            "clinician_evaluation": {"review_status": "draft_pending_clinician_review", "is_clinician_approved": False}
        }
        # Claim asserts KF rings and 95%
        reply = "Kayser-Fleischer rings are present in over 95% of patients with neurological Wilson's Disease [AASLD WD Guidelines.pdf, p. 5]."
        # Truncated passage contains only the first sentence about tremor, omitting KF rings
        truncated_passage = {
            "header": "[Source: AASLD WD Guidelines.pdf | Page: 5]",
            "preview": "Initial manifestations include tremors and dysarthria.",
            "is_truncated": True
        }
        tel = {"retrieved_passages": [truncated_passage], "evidence_coverage": {}}

        res = self.evaluator.evaluate_case(case, rag_reply=reply, telemetry=tel)

        det = res["details"]["claim_level_support_details"]
        item = det["details"][0]
        # Missing evidence must NOT become a positive support result
        self.assertFalse(item["is_supported"])
        self.assertTrue(item["is_truncated"])
        self.assertIn("truncated_passage_evidence_insufficient", item["reason"])


if __name__ == "__main__":
    unittest.main()

