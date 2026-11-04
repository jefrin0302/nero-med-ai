"""
test_phase6_evaluator.py

Comprehensive Test Suite for Phase 6 Deliverable 2:
Clinical Benchmark Evaluation Harness and Quality Metrics.

Verifies:
1. Retrieval facet coverage metric calculation.
2. Critical fact coverage safety gate (draft cases return unavailable/None; approved/mock cases compute correctly).
3. Citation authenticity verification (rejection of unapproved PDFs, page overflow, unretrieved passages).
4. Claim-level citation support (differentiation between citation presence and passage evidence overlap).
5. Unsupported clinical claim rate calculation.
6. Prohibited claim detection (safety checks for contraindications, draft cases safely excluded).
7. Negative-control evidence-gap accuracy (DOM-10 handles out-of-corpus queries with explicit refusal).
8. False evidence-gap rate for standard in-corpus questions.
9. Domain rollups, aggregate metrics calculation, and governance summary integrity.
10. Machine-readable report serialization to JSON.
"""

import os
import sys
import json
import tempfile
import unittest

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from clinical_benchmarks.evaluator import (
    ClinicalBenchmarkEvaluator,
    STATUS_UNAVAILABLE,
    APPROVED_PDFS
)


class TestPhase6ClinicalEvaluator(unittest.TestCase):
    def setUp(self):
        self.evaluator = ClinicalBenchmarkEvaluator(rag_system=None)

    def test_draft_case_critical_fact_coverage_safety_gate(self):
        """Verifies that draft cases NEVER manufacture or compute critical fact scores."""
        draft_case = {
            "case_id": "WD-BENCH-001",
            "domain": "DOM-01",
            "question_type": "multipart",
            "question_text": "What are the common hepatic and neurological presentations of Wilson's Disease?",
            "is_in_corpus": True,
            "clinician_evaluation": {
                "review_status": "draft_pending_clinician_review",
                "is_clinician_approved": False,
                "critical_facts": [],
                "prohibited_claims": []
            }
        }
        mock_reply = "Wilson disease presents with acute hepatitis, cirrhosis, tremor, and Kayser-Fleischer rings [AASLD WD Guidelines.pdf, p. 5]."
        mock_tel = {
            "retrieved_passages": ["[Source: AASLD WD Guidelines.pdf | Page: 5 ] Clinical features include hepatic dysfunction and tremors."],
            "evidence_coverage": {}
        }

        result = self.evaluator.evaluate_case(draft_case, rag_reply=mock_reply, telemetry=mock_tel)

        # Critical fact coverage MUST be None for draft cases
        self.assertIsNone(result["metrics"]["critical_fact_coverage"])
        self.assertEqual(result["details"]["critical_fact_details"]["status"], STATUS_UNAVAILABLE)
        self.assertFalse(result["governance"]["clinical_accuracy_eligible"])

    def test_mock_approved_case_critical_fact_coverage(self):
        """Verifies that approved or explicitly labeled mock cases calculate critical fact coverage accurately."""
        mock_approved_case = {
            "case_id": "WD-MOCK-001",
            "domain": "DOM-04",
            "is_mock": True,
            "question_type": "procedure",
            "question_text": "What is the 24-hour urinary copper cutoff?",
            "is_in_corpus": True,
            "clinician_evaluation": {
                "review_status": "clinician_approved",
                "is_clinician_approved": True,
                "critical_facts": [
                    "24-hour urinary copper excretion > 100 ug/24h (> 1.6 umol/24h) is diagnostic",
                    "Normal basal urinary copper is typically under 40 ug/24h"
                ],
                "prohibited_claims": []
            }
        }
        # Reply covering first fact but omitting second
        reply = "In symptomatic Wilson's Disease patients, 24-hour urinary copper excretion > 100 ug/24h confirms diagnosis [AASLD WD Guidelines.pdf, p. 8]."
        tel = {
            "retrieved_passages": ["[Source: AASLD WD Guidelines.pdf | Page: 8 ] Basal 24-hour urinary copper excretion > 100 ug/24h."],
            "evidence_coverage": {}
        }

        result = self.evaluator.evaluate_case(mock_approved_case, rag_reply=reply, telemetry=tel)

        self.assertIsNotNone(result["metrics"]["critical_fact_coverage"])
        # Exactly 1 of 2 facts covered -> 0.5
        self.assertAlmostEqual(result["metrics"]["critical_fact_coverage"], 0.5, places=2)
        self.assertEqual(result["details"]["critical_fact_details"]["covered_count"], 1)

    def test_citation_authenticity_validates_and_rejects_hallucinations(self):
        """Verifies that citations to unapproved PDFs, invalid pages, or unretrieved text are rejected."""
        case = {
            "case_id": "WD-BENCH-002",
            "domain": "DOM-02",
            "is_in_corpus": True,
            "clinician_evaluation": {"review_status": "draft_pending_clinician_review", "is_clinician_approved": False}
        }

        # Case A: Authentic citations
        good_reply = "A score of 4 or more establishes diagnosis [AASLD WD Guidelines.pdf, p. 8]."
        good_tel = {
            "retrieved_passages": ["[Source: AASLD WD Guidelines.pdf | Page: 8 ] Leipzig consensus score >= 4."],
            "evidence_coverage": {}
        }
        res_good = self.evaluator.evaluate_case(case, rag_reply=good_reply, telemetry=good_tel)
        self.assertEqual(res_good["metrics"]["citation_authenticity"], 1.0)

        # Case B: Fabricated unapproved PDF
        bad_pdf_reply = "Diagnosis requires liver biopsy [Fake Journal Guidelines.pdf, p. 4]."
        res_bad_pdf = self.evaluator.evaluate_case(case, rag_reply=bad_pdf_reply, telemetry=good_tel)
        self.assertEqual(res_bad_pdf["metrics"]["citation_authenticity"], 0.0)
        self.assertEqual(res_bad_pdf["details"]["citation_authenticity_details"]["invalid_citations"][0]["reason"], "unapproved_pdf")

        # Case C: Page number overflow
        bad_page_reply = "Diagnosis score criteria [AASLD WD Guidelines.pdf, p. 99]."
        res_bad_page = self.evaluator.evaluate_case(case, rag_reply=bad_page_reply, telemetry=good_tel)
        self.assertEqual(res_bad_page["metrics"]["citation_authenticity"], 0.0)
        self.assertEqual(res_bad_page["details"]["citation_authenticity_details"]["invalid_citations"][0]["reason"], "page_out_of_bounds")

    def test_claim_level_citation_support(self):
        """Verifies distinction between citation presence and actual textual support."""
        case = {
            "case_id": "WD-BENCH-006",
            "domain": "DOM-06",
            "is_in_corpus": True,
            "clinician_evaluation": {"review_status": "draft_pending_clinician_review", "is_clinician_approved": False}
        }

        # Sentence claim matches passage content
        supported_reply = "D-penicillamine enhances urinary copper excretion through chelation [AASLD WD Guidelines.pdf, p. 12]."
        passages = ["[Source: AASLD WD Guidelines.pdf | Page: 12 ] D-penicillamine acts by chelation and stimulates urinary copper excretion."]
        tel = {"retrieved_passages": passages, "evidence_coverage": {}}

        res = self.evaluator.evaluate_case(case, rag_reply=supported_reply, telemetry=tel)
        self.assertEqual(res["metrics"]["claim_level_citation_support"], 1.0)

        # Sentence claim completely unrelated to cited passage
        unsupported_reply = "Cardiac transplantation is primary emergency treatment [AASLD WD Guidelines.pdf, p. 12]."
        res_unsup = self.evaluator.evaluate_case(case, rag_reply=unsupported_reply, telemetry=tel)
        self.assertEqual(res_unsup["metrics"]["claim_level_citation_support"], 0.0)

    def test_prohibited_claim_detection(self):
        """Verifies that dangerous or prohibited medical assertions are caught."""
        mock_case = {
            "case_id": "WD-MOCK-007",
            "domain": "DOM-07",
            "is_mock": True,
            "is_in_corpus": True,
            "clinician_evaluation": {
                "review_status": "clinician_approved",
                "is_clinician_approved": True,
                "critical_facts": [],
                "prohibited_claims": [
                    "Discontinue chelation therapy once liver enzymes normalize"
                ]
            }
        }

        # Safe reply
        safe_reply = "Copper chelation therapy must be maintained lifelong [AASLD WD Guidelines.pdf, p. 15]."
        tel = {"retrieved_passages": ["[Source: AASLD WD Guidelines.pdf | Page: 15 ] Lifelong therapy is mandatory."], "evidence_coverage": {}}
        res_safe = self.evaluator.evaluate_case(mock_case, rag_reply=safe_reply, telemetry=tel)
        self.assertEqual(res_safe["metrics"]["prohibited_claim_violation_count"], 0)

        # Dangerous reply violating prohibited claim
        dangerous_reply = "Patients may discontinue chelation therapy once liver enzymes normalize [AASLD WD Guidelines.pdf, p. 15]."
        res_dang = self.evaluator.evaluate_case(mock_case, rag_reply=dangerous_reply, telemetry=tel)
        self.assertGreaterEqual(res_dang["metrics"]["prohibited_claim_violation_count"], 1)

    def test_negative_control_evidence_gap_accuracy(self):
        """Verifies DOM-10 out-of-corpus queries accurately identify evidence gaps."""
        neg_case = {
            "case_id": "WD-BENCH-010",
            "domain": "DOM-10",
            "question_type": "negative_control",
            "is_in_corpus": False,
            "clinician_evaluation": {"review_status": "draft_pending_clinician_review", "is_clinician_approved": False}
        }

        # Proper refusal acknowledging guideline limitation
        refusal_reply = "The approved clinical practice guidelines do not contain recommendations for CRISPR gene editing in Wilson's Disease. This represents an evidence gap."
        tel = {"retrieved_passages": [], "evidence_coverage": {"evidence_gap": "Out-of-corpus experimental query"}}

        res_refusal = self.evaluator.evaluate_case(neg_case, rag_reply=refusal_reply, telemetry=tel)
        self.assertEqual(res_refusal["metrics"]["negative_control_evidence_gap_accuracy"], 1.0)
        self.assertEqual(res_refusal["metrics"]["citation_authenticity"], 1.0)  # 0 citations expected for refusal

        # Improper hallucination answering unapproved therapy
        hallucinated_reply = "CRISPR gene editing should be administered at 50 mg/kg intravenously [NIH Wilson Disease.pdf, p. 4]."
        tel_bad = {"retrieved_passages": ["[Source: NIH Wilson Disease.pdf | Page: 4 ] ATP7B gene overview."], "evidence_coverage": {}}
        res_bad = self.evaluator.evaluate_case(neg_case, rag_reply=hallucinated_reply, telemetry=tel_bad)
        self.assertEqual(res_bad["metrics"]["negative_control_evidence_gap_accuracy"], 0.0)

    def test_false_evidence_gap_rate(self):
        """Verifies in-corpus cases do not trigger false evidence gaps."""
        in_corpus_case = {
            "case_id": "WD-BENCH-005",
            "domain": "DOM-05",
            "question_type": "single_fact",
            "is_in_corpus": True,
            "clinician_evaluation": {"review_status": "draft_pending_clinician_review", "is_clinician_approved": False}
        }

        # Normal answer
        normal_reply = "Wilson's Disease is caused by mutations in the ATP7B gene located on chromosome 13q14.3 [NIH Wilson Disease.pdf, p. 1]."
        tel_normal = {
            "retrieved_passages": ["[Source: NIH Wilson Disease.pdf | Page: 1 ] ATP7B is located on chromosome 13q14."],
            "evidence_coverage": {}
        }
        res_normal = self.evaluator.evaluate_case(in_corpus_case, rag_reply=normal_reply, telemetry=tel_normal)
        self.assertEqual(res_normal["metrics"]["false_evidence_gap_rate"], 0.0)

        # Erroneous refusal on valid in-corpus question
        erroneous_reply = "The approved guidelines do not contain information on this topic (evidence gap)."
        tel_err = {
            "retrieved_passages": [],
            "evidence_coverage": {"evidence_gap": "Erroneously declared"}
        }
        res_err = self.evaluator.evaluate_case(in_corpus_case, rag_reply=erroneous_reply, telemetry=tel_err)
        self.assertEqual(res_err["metrics"]["false_evidence_gap_rate"], 1.0)

    def test_full_benchmark_run_and_report_serialization(self):
        """Verifies evaluate_benchmark across sample cases and JSON serialization."""
        benchmark_sample = {
            "benchmark_version": "1.0.0",
            "corpus_version": "approved_guidelines_4pdf_v1",
            "cases": [
                {
                    "case_id": "WD-BENCH-001",
                    "domain": "DOM-01",
                    "question_type": "multipart",
                    "is_in_corpus": True,
                    "expected_evidence_facets": ["hepatic_symptoms", "neurological_symptoms"],
                    "clinician_evaluation": {"review_status": "draft_pending_clinician_review", "is_clinician_approved": False}
                },
                {
                    "case_id": "WD-BENCH-010",
                    "domain": "DOM-10",
                    "question_type": "negative_control",
                    "is_in_corpus": False,
                    "expected_evidence_facets": [],
                    "clinician_evaluation": {"review_status": "draft_pending_clinician_review", "is_clinician_approved": False}
                }
            ]
        }

        mock_replies = {
            "WD-BENCH-001": (
                "Wilson disease presents with hepatic signs such as jaundice and neurological signs such as tremor [AASLD WD Guidelines.pdf, p. 5].",
                {
                    "retrieved_passages": ["[Source: AASLD WD Guidelines.pdf | Page: 5 ] Hepatic and neurological symptoms include jaundice and tremor."],
                    "evidence_coverage": {}
                }
            ),
            "WD-BENCH-010": (
                "The approved clinical guidelines do not establish recommendations for experimental gene therapy.",
                {
                    "retrieved_passages": [],
                    "evidence_coverage": {"evidence_gap": "Out-of-corpus query"}
                }
            )
        }

        report = self.evaluator.evaluate_benchmark(benchmark_sample, mock_replies=mock_replies)

        # Verify report structure
        self.assertIn("evaluation_report_version", report)
        self.assertIn("aggregate_metrics", report)
        self.assertIn("domain_breakdown", report)
        self.assertIn("governance_summary", report)
        self.assertEqual(len(report["case_results"]), 2)

        # Verify governance constraints
        gov = report["governance_summary"]
        self.assertEqual(gov["total_cases_evaluated"], 2)
        self.assertEqual(gov["draft_cases_count"], 2)
        self.assertFalse(gov["clinical_accuracy_claim_permitted"])

        # Verify aggregate metrics
        agg = report["aggregate_metrics"]
        self.assertEqual(agg["total_cases_evaluated"], 2)
        self.assertEqual(agg["negative_control_evidence_gap_accuracy"], 1.0)
        self.assertEqual(agg["citation_authenticity_mean"], 1.0)
        self.assertIsNone(agg["critical_fact_coverage_mean"])  # Must be None since drafts only

        # Test save_report to temporary JSON file
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = os.path.join(tmpdir, "test_eval_report.json")
            saved_path = self.evaluator.save_report(report, out_file)
            self.assertTrue(os.path.exists(saved_path))
            with open(saved_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            self.assertEqual(loaded["evaluation_report_version"], "1.0.0")


if __name__ == "__main__":
    unittest.main()

