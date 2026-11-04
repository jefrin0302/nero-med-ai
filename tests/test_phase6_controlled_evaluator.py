"""
test_phase6_controlled_evaluator.py

NeuroMed AI — Controlled Evaluator Reliability Test Suite.
Phase 6 Step 2.1 Audit Deliverable.

Evaluates 10 controlled mock scenarios to audit evaluator precision, false positives,
and false negatives across citation support, authenticity, and numerical grounding.

IMPORTANT:
These are synthetic, explicitly labeled mock cases (is_mock=True).
They are NOT clinical benchmark ground truth and MUST NOT be counted
towards real clinical accuracy performance.
"""

import os
import sys
import unittest

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from clinical_benchmarks.evaluator import ClinicalBenchmarkEvaluator, STATUS_UNAVAILABLE


class TestPhase6ControlledEvaluatorScenarios(unittest.TestCase):
    def setUp(self):
        self.evaluator = ClinicalBenchmarkEvaluator(rag_system=None)

    def test_scenario_01_directly_supporting_passage_and_correct_citation(self):
        """
        Scenario 1: A clinical claim with a directly supporting passage and correct citation.
        Expected: Citation Authenticity = 1.0, Claim Support = 1.0 (Supported, complete),
                  Unsupported Claim Rate = 0.0.
        """
        case = {
            "case_id": "MOCK-SCENARIO-01",
            "domain": "DOM-01",
            "is_mock": True,
            "is_in_corpus": True,
            "clinician_evaluation": {"review_status": "draft_pending_clinician_review", "is_clinician_approved": False}
        }
        reply = "Kayser-Fleischer rings are detected in over 95% of patients with neurological Wilson's Disease [AASLD WD Guidelines.pdf, p. 5]."
        passages = ["[Source: AASLD WD Guidelines.pdf | Page: 5 ] Kayser-Fleischer rings are detected in greater than 95% of patients presenting with neurological symptoms of Wilson disease."]
        tel = {"retrieved_passages": passages, "evidence_coverage": {}}

        res = self.evaluator.evaluate_case(case, rag_reply=reply, telemetry=tel)

        self.assertEqual(res["metrics"]["citation_authenticity"], 1.0)
        self.assertEqual(res["metrics"]["claim_level_citation_support"], 1.0)
        self.assertEqual(res["metrics"]["unsupported_clinical_claim_rate"], 0.0)
        detail = res["details"]["claim_level_support_details"]["details"][0]
        self.assertTrue(detail["is_supported"])
        self.assertEqual(detail["support_level"], "complete")

    def test_scenario_02_valid_citation_pointing_to_unrelated_passage(self):
        """
        Scenario 2: A valid citation pointing to an unrelated passage.
        Demonstrates that Citation Authenticity does NOT imply Claim Support.
        Expected: Citation Authenticity = 1.0, Claim Support = 0.0 (Unsupported),
                  Unsupported Claim Rate = 1.0.
        """
        case = {
            "case_id": "MOCK-SCENARIO-02",
            "domain": "DOM-03",
            "is_mock": True,
            "is_in_corpus": True,
            "clinician_evaluation": {"review_status": "draft_pending_clinician_review", "is_clinician_approved": False}
        }
        reply = "Serum ceruloplasmin is markedly reduced in 90% of hepatic presentations [AASLD WD Guidelines.pdf, p. 12]."
        # Passage on page 12 discusses penicillamine dosing, NOT ceruloplasmin!
        passages = ["[Source: AASLD WD Guidelines.pdf | Page: 12 ] D-penicillamine dosing should start at 250-500 mg daily and titrate upwards based on urinary copper excretion."]
        tel = {"retrieved_passages": passages, "evidence_coverage": {}}

        res = self.evaluator.evaluate_case(case, rag_reply=reply, telemetry=tel)

        # Authenticity is 1.0 because AASLD p. 12 is an approved PDF within page bounds in the retrieved pool
        self.assertEqual(res["metrics"]["citation_authenticity"], 1.0)
        # But claim support is 0.0 because the passage text is completely unrelated to the claim!
        self.assertEqual(res["metrics"]["claim_level_citation_support"], 0.0)
        self.assertEqual(res["metrics"]["unsupported_clinical_claim_rate"], 1.0)
        detail = res["details"]["claim_level_support_details"]["details"][0]
        self.assertFalse(detail["is_supported"])
        self.assertEqual(detail["support_level"], "unsupported")

    def test_scenario_03_claim_partially_supported_by_passage(self):
        """
        Scenario 3: A claim partially supported by its passage.
        Expected: Claim Support != 1.0 (is_supported=False for complete support, flagged as partial).
        """
        case = {
            "case_id": "MOCK-SCENARIO-03",
            "domain": "DOM-04",
            "is_mock": True,
            "is_in_corpus": True,
            "clinician_evaluation": {"review_status": "draft_pending_clinician_review", "is_clinician_approved": False}
        }
        reply = "Diagnostic evaluation requires 24-hour urinary copper measurement and immediate mandatory liver biopsy in all suspected individuals [AASLD WD Guidelines.pdf, p. 8]."
        # Passage supports urinary copper, but liver biopsy is optional/reserve, not mandatory for all
        passages = ["[Source: AASLD WD Guidelines.pdf | Page: 8 ] Diagnostic evaluation includes 24-hour urinary copper measurement. Liver biopsy is reserved for cases where noninvasive tests remain inconclusive."]
        tel = {"retrieved_passages": passages, "evidence_coverage": {}}

        res = self.evaluator.evaluate_case(case, rag_reply=reply, telemetry=tel)

        detail = res["details"]["claim_level_support_details"]["details"][0]
        # Partial support is detected, but not treated as complete support
        self.assertTrue(detail["is_partial"])
        self.assertFalse(detail["is_supported"])
        self.assertEqual(detail["support_level"], "partial")

    def test_scenario_04_numerical_threshold_absent_from_cited_passage(self):
        """
        Scenario 4: A numerical threshold absent from the cited passage.
        Expected: Claim Support = 0.0 (Unsupported due to missing numerical threshold '250').
        """
        case = {
            "case_id": "MOCK-SCENARIO-04",
            "domain": "DOM-04",
            "is_mock": True,
            "is_in_corpus": True,
            "clinician_evaluation": {"review_status": "draft_pending_clinician_review", "is_clinician_approved": False}
        }
        # The claim states 250 ug/24h, whereas the passage states 100 ug/24h
        reply = "Basal 24-hour urinary copper excretion exceeding 250 ug/24h is the diagnostic cutoff [AASLD WD Guidelines.pdf, p. 8]."
        passages = ["[Source: AASLD WD Guidelines.pdf | Page: 8 ] Basal 24-hour urinary copper excretion exceeding 100 ug/24h (1.6 umol/24h) is diagnostic of Wilson disease."]
        tel = {"retrieved_passages": passages, "evidence_coverage": {}}

        res = self.evaluator.evaluate_case(case, rag_reply=reply, telemetry=tel)

        self.assertEqual(res["metrics"]["claim_level_citation_support"], 0.0)
        detail = res["details"]["claim_level_support_details"]["details"][0]
        self.assertFalse(detail["is_supported"])
        self.assertIn("unsupported_numerical_threshold", detail["reason"])
        self.assertIn("250", detail["reason"])

    def test_scenario_05_fabricated_source_filename(self):
        """
        Scenario 5: A fabricated source filename.
        Expected: Citation Authenticity = 0.0 (unapproved_pdf), Claim Support = 0.0.
        """
        case = {
            "case_id": "MOCK-SCENARIO-05",
            "domain": "DOM-05",
            "is_mock": True,
            "is_in_corpus": True,
            "clinician_evaluation": {"review_status": "draft_pending_clinician_review", "is_clinician_approved": False}
        }
        reply = "ATP7B gene mutations are identified in chromosome 13 [New England Journal Wilson Guidelines.pdf, p. 4]."
        passages = ["[Source: AASLD WD Guidelines.pdf | Page: 8 ] ATP7B mutations confirm Wilson's disease."]
        tel = {"retrieved_passages": passages, "evidence_coverage": {}}

        res = self.evaluator.evaluate_case(case, rag_reply=reply, telemetry=tel)

        self.assertEqual(res["metrics"]["citation_authenticity"], 0.0)
        self.assertEqual(res["metrics"]["claim_level_citation_support"], 0.0)
        auth_details = res["details"]["citation_authenticity_details"]
        self.assertEqual(auth_details["invalid_citations"][0]["reason"], "unapproved_pdf")

    def test_scenario_06_real_approved_pdf_with_unretrieved_page(self):
        """
        Scenario 6: A real approved PDF with an unretrieved page.
        Expected: Citation Authenticity = 0.0 (unretrieved_passage), Claim Support = 0.0.
        """
        case = {
            "case_id": "MOCK-SCENARIO-06",
            "domain": "DOM-06",
            "is_mock": True,
            "is_in_corpus": True,
            "clinician_evaluation": {"review_status": "draft_pending_clinician_review", "is_clinician_approved": False}
        }
        # Cites page 14 of AASLD, but only page 5 was retrieved
        reply = "Trientine is an effective chelator with lower neurological deterioration [AASLD WD Guidelines.pdf, p. 14]."
        passages = ["[Source: AASLD WD Guidelines.pdf | Page: 5 ] Kayser-Fleischer rings presentation."]
        tel = {"retrieved_passages": passages, "evidence_coverage": {}}

        res = self.evaluator.evaluate_case(case, rag_reply=reply, telemetry=tel)

        self.assertEqual(res["metrics"]["citation_authenticity"], 0.0)
        self.assertEqual(res["metrics"]["claim_level_citation_support"], 0.0)
        auth_details = res["details"]["citation_authenticity_details"]
        self.assertEqual(auth_details["invalid_citations"][0]["reason"], "unretrieved_passage")

    def test_scenario_07_two_claims_with_separate_citations(self):
        """
        Scenario 7: Two claims with separate citations.
        Expected: 2 cited sentences, 2 authentic citations, 2 supported claims -> Score 1.0.
        """
        case = {
            "case_id": "MOCK-SCENARIO-07",
            "domain": "DOM-06",
            "is_mock": True,
            "is_in_corpus": True,
            "clinician_evaluation": {"review_status": "draft_pending_clinician_review", "is_clinician_approved": False}
        }
        reply = (
            "Kayser-Fleischer rings reflect copper accumulation in the cornea [AASLD WD Guidelines.pdf, p. 5]. "
            "D-penicillamine initial dose is 250-500 mg daily [EASL-ERN Clinical Practice Guidelines on Wilson's disease.pdf, p. 12]."
        )
        passages = [
            "[Source: AASLD WD Guidelines.pdf | Page: 5 ] Kayser-Fleischer rings reflect copper accumulation in the cornea.",
            "[Source: EASL-ERN Clinical Practice Guidelines on Wilson's disease.pdf | Page: 12 ] D-penicillamine initial dose is 250-500 mg daily."
        ]
        tel = {"retrieved_passages": passages, "evidence_coverage": {}}

        res = self.evaluator.evaluate_case(case, rag_reply=reply, telemetry=tel)

        self.assertEqual(res["metrics"]["citation_authenticity"], 1.0)
        self.assertEqual(res["metrics"]["claim_level_citation_support"], 1.0)
        self.assertEqual(res["details"]["claim_level_support_details"]["total_cited_sentences"], 2)
        self.assertEqual(res["details"]["claim_level_support_details"]["supported_count"], 2)

    def test_scenario_08_one_claim_supported_by_multiple_passages(self):
        """
        Scenario 8: One claim supported by multiple passages.
        Expected: Both citations authentic, claim supported -> Score 1.0.
        """
        case = {
            "case_id": "MOCK-SCENARIO-08",
            "domain": "DOM-05",
            "is_mock": True,
            "is_in_corpus": True,
            "clinician_evaluation": {"review_status": "draft_pending_clinician_review", "is_clinician_approved": False}
        }
        reply = "Wilson's Disease is inherited in an autosomal recessive pattern due to ATP7B mutations [AASLD WD Guidelines.pdf, p. 3] [NIH Wilson Disease.pdf, p. 1]."
        passages = [
            "[Source: AASLD WD Guidelines.pdf | Page: 3 ] Wilson disease is an autosomal recessive disorder of copper metabolism.",
            "[Source: NIH Wilson Disease.pdf | Page: 1 ] ATP7B gene mutations cause autosomal recessive Wilson disease."
        ]
        tel = {"retrieved_passages": passages, "evidence_coverage": {}}

        res = self.evaluator.evaluate_case(case, rag_reply=reply, telemetry=tel)

        self.assertEqual(res["metrics"]["citation_authenticity"], 1.0)
        self.assertEqual(res["metrics"]["claim_level_citation_support"], 1.0)
        self.assertEqual(res["details"]["claim_level_support_details"]["total_cited_sentences"], 1)

    def test_scenario_09_paraphrased_claim_preserving_passage_meaning(self):
        """
        Scenario 9: A paraphrased claim that preserves the meaning of the passage.
        Expected: Claim Support = 1.0 (Supported via substantive term overlap and synonym recognition).
        """
        case = {
            "case_id": "MOCK-SCENARIO-09",
            "domain": "DOM-07",
            "is_mock": True,
            "is_in_corpus": True,
            "clinician_evaluation": {"review_status": "draft_pending_clinician_review", "is_clinician_approved": False}
        }
        # Paraphrased phrasing using "co-administered vitamin B6" vs "Supplemental pyridoxine"
        reply = "Patients should take supplemental vitamin B6 with penicillamine to counteract antipyridoxine adverse effects [AASLD WD Guidelines.pdf, p. 13]."
        passages = ["[Source: AASLD WD Guidelines.pdf | Page: 13 ] Supplemental pyridoxine (vitamin B6, 25 to 50 mg daily) is recommended with D-penicillamine because of antipyridoxine effects."]
        tel = {"retrieved_passages": passages, "evidence_coverage": {}}

        res = self.evaluator.evaluate_case(case, rag_reply=reply, telemetry=tel)

        self.assertEqual(res["metrics"]["claim_level_citation_support"], 1.0)
        detail = res["details"]["claim_level_support_details"]["details"][0]
        self.assertTrue(detail["is_supported"])
        self.assertEqual(detail["support_level"], "complete")

    def test_scenario_10_claim_contradicted_by_cited_passage(self):
        """
        Scenario 10: A claim contradicted by the cited passage.
        Expected: Prohibited claim violation detected, Claim Support = 0.0 (Contradicted).
        """
        case = {
            "case_id": "MOCK-SCENARIO-10",
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
        # The claim dangerously advises discontinuing chelation
        reply = "Patients may discontinue chelation therapy once liver enzymes normalize [AASLD WD Guidelines.pdf, p. 15]."
        passages = ["[Source: AASLD WD Guidelines.pdf | Page: 15 ] Lifelong therapy is mandatory. Discontinuation of treatment almost invariably results in fatal hepatic relapse."]
        tel = {"retrieved_passages": passages, "evidence_coverage": {}}

        res = self.evaluator.evaluate_case(case, rag_reply=reply, telemetry=tel)

        # Prohibited claim violation must be detected
        self.assertGreaterEqual(res["metrics"]["prohibited_claim_violation_count"], 1)
        # Contradicted claim must fail claim support
        self.assertEqual(res["metrics"]["claim_level_citation_support"], 0.0)
        detail = res["details"]["claim_level_support_details"]["details"][0]
        self.assertFalse(detail["is_supported"])
        self.assertIn("contradicted", detail["reason"])


if __name__ == "__main__":
    unittest.main()

