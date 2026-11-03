"""
test_phase3_coverage.py

Unit and Integration Test Suite for Phase 3:
Evidence Coverage Checking and Diagnostic Evidence Verification.

Validates:
1. Complete evidence coverage across clinical facets.
2. Identification of partially missing evidence facets.
3. Targeted second-pass retrieval activation and candidate pool expansion.
4. Explicit evidence gap reporting when guidelines lack data.
5. Simple / atomic clinical questions are never incorrectly marked incomplete.
6. Preservation of source document, page number, and passage provenance.
7. Telemetry completeness: facets_expected, facets_covered, facets_missing,
   coverage_percentage, second_pass_performed, and is_complete.
8. Standalone compatibility without external network dependencies.
"""

import os
import unittest
import re
from rag_system import (
    MedicalRAGSystem,
    decompose_clinical_query,
    APPROVED_PDFS
)


class TestPhase3EvidenceCoverage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.system = MedicalRAGSystem()
        cls.system.gemini_client = None  # Force offline evaluation
        cls.system._ensure_documents_loaded()

    def setUp(self):
        self.system.gemini_client = None

    # =========================================================================
    # 1. COMPLETE EVIDENCE COVERAGE
    # =========================================================================
    def test_complete_evidence_coverage_urinary_copper(self):
        """
        Verifies that for a 24-hour urinary copper query, full candidate retrieval
        covers all 4 required facets: collection, measurement, thresholds, and monitoring.
        """
        query = "How is 24-hour urinary copper collected, measured, and interpreted?"
        subqueries = decompose_clinical_query(query)
        candidates, _, _ = self.system._retrieve_hybrid_candidates(query, subqueries)

        updated_candidates, coverage = self.system.check_retrieval_coverage(query, subqueries, candidates)

        self.assertIn("facets_expected", coverage)
        self.assertIn("facets_covered", coverage)
        self.assertIn("facets_missing", coverage)
        self.assertIn("coverage_percentage", coverage)
        self.assertIn("is_complete", coverage)

        # Expected facets for urinary copper
        expected = ["collection_procedure", "laboratory_measurement", "diagnostic_thresholds", "treatment_monitoring"]
        for f in expected:
            self.assertIn(f, coverage["facets_expected"])

        # With approved guidelines, all 4 facets are covered
        self.assertTrue(coverage["is_complete"], f"Expected complete coverage, got missing: {coverage.get('facets_missing')}")
        self.assertEqual(len(coverage["facets_missing"]), 0)
        self.assertEqual(coverage["coverage_percentage"], 100.0)
        self.assertIsNone(coverage.get("evidence_gap"))

    def test_complete_evidence_coverage_treatment_comparison(self):
        """
        Verifies that treatment comparison covers D-penicillamine, trientine, and zinc.
        """
        query = "What is the difference between D-Penicillamine, Trientine, and Zinc in Wilson disease?"
        subqueries = decompose_clinical_query(query)
        candidates, _, _ = self.system._retrieve_hybrid_candidates(query, subqueries)

        updated_candidates, coverage = self.system.check_retrieval_coverage(query, subqueries, candidates)

        expected = ["d_penicillamine", "trientine", "zinc_therapy"]
        for f in expected:
            self.assertIn(f, coverage["facets_expected"])

        self.assertTrue(coverage["is_complete"])
        self.assertEqual(len(coverage["facets_missing"]), 0)
        self.assertEqual(coverage["coverage_percentage"], 100.0)

    def test_complete_evidence_coverage_leipzig_scoring(self):
        """
        Verifies that Leipzig scoring criteria query covers clinical/KF, biochemical, hepatic, and genetics facets.
        """
        query = "What are the criteria and points in the Leipzig scoring system for Wilson's disease?"
        subqueries = decompose_clinical_query(query)
        candidates, _, _ = self.system._retrieve_hybrid_candidates(query, subqueries)

        updated_candidates, coverage = self.system.check_retrieval_coverage(query, subqueries, candidates)

        expected = ["clinical_and_kf_rings", "biochemical_copper_and_ceruloplasmin", "hepatic_copper_or_histology", "atp7b_genetics_and_cutoff"]
        for f in expected:
            self.assertIn(f, coverage["facets_expected"])

        self.assertTrue(coverage["is_complete"])
        self.assertEqual(coverage["coverage_percentage"], 100.0)

    # =========================================================================
    # 2. PARTIALLY MISSING EVIDENCE IDENTIFICATION
    # =========================================================================
    def test_partially_missing_evidence_detection(self):
        """
        Verifies that if candidate pool lacks specific facets, check_retrieval_coverage
        accurately identifies exactly which facets are missing.
        """
        query = "How is 24-hour urinary copper collected, measured, and interpreted?"
        subqueries = decompose_clinical_query(query)

        # Deliberately filter candidate pool to only passages containing 'penicillamine' (monitoring only)
        # and exclude collection/measurement passages
        restricted_candidates = set()
        for idx, (p, h, b, b_low) in enumerate(self.system._parsed_passages):
            if "penicillamine" in b_low and not any(w in b_low for w in ["polyethylene", "acid", "washing", "spectrophotometry"]):
                restricted_candidates.add(idx)
                if len(restricted_candidates) >= 3:
                    break

        # Temporarily mock _bm25_search and _semantic_search to not find missing facets during test
        original_bm25 = self.system._bm25_search
        original_semantic = self.system._semantic_search
        try:
            self.system._bm25_search = lambda q, k=5: []
            self.system._semantic_search = lambda q, k=5: []

            _, coverage = self.system.check_retrieval_coverage(query, subqueries, set(restricted_candidates))

            self.assertFalse(coverage["is_complete"])
            self.assertGreater(len(coverage["facets_missing"]), 0)
            self.assertLess(coverage["coverage_percentage"], 100.0)
            self.assertIsNotNone(coverage["evidence_gap"])
            self.assertTrue(coverage["second_pass_performed"])
        finally:
            self.system._bm25_search = original_bm25
            self.system._semantic_search = original_semantic

    # =========================================================================
    # 3. TARGETED SECOND-PASS RETRIEVAL ACTIVATION
    # =========================================================================
    def test_targeted_second_pass_retrieval_recovers_missing_facets(self):
        """
        Verifies that when initial candidate pool has missing facets, second-pass
        retrieval triggers targeted search queries and successfully expands the candidates.
        """
        query = "How is 24-hour urinary copper collected, measured, and interpreted?"
        subqueries = decompose_clinical_query(query)

        # Start with an artificially sparse candidate set (1 passage only)
        sparse_candidates = {0}
        initial_count = len(sparse_candidates)

        updated_candidates, coverage = self.system.check_retrieval_coverage(query, subqueries, set(sparse_candidates))

        # Second pass must have run and added new candidates
        self.assertTrue(coverage["second_pass_performed"])
        self.assertGreater(len(updated_candidates), initial_count)
        # Should have recovered missing facets
        self.assertGreater(len(coverage["facets_covered"]), 0)

    # =========================================================================
    # 4. EVIDENCE GAP REPORTING
    # =========================================================================
    def test_evidence_gap_reported_for_out_of_corpus_facet(self):
        """
        Verifies that when guideline evidence is genuinely unavailable, an explicit
        evidence gap is reported without assuming or fabricating data.
        """
        query = "What is the recommended dose of remdesivir in COVID-19?"
        reply, telemetry = self.system.ask_chatbot_with_telemetry(query)

        # Telemetry evidence coverage should reflect out-of-corpus gap
        coverage = telemetry.get("evidence_coverage", {})
        self.assertFalse(coverage.get("is_complete", False))
        self.assertIn("direct_guideline_evidence", coverage.get("facets_missing", []))
        self.assertIsNotNone(coverage.get("evidence_gap"))

        # The response must honestly report lack of evidence and NEVER fabricate dosages
        self.assertIn("do not contain sufficient evidence", reply.lower())
        self.assertNotIn("100 mg", reply)
        self.assertNotIn("200 mg", reply)

    # =========================================================================
    # 5. SIMPLE QUESTIONS NOT INCORRECTLY MARKED INCOMPLETE
    # =========================================================================
    def test_simple_atomic_queries_not_marked_incomplete(self):
        """
        Verifies that simple atomic questions (e.g. gene, KF rings, ceruloplasmin definition)
        are NOT incorrectly marked incomplete when direct guideline evidence exists.
        """
        simple_queries = [
            "Which gene is responsible for Wilson's Disease?",
            "What are Kayser-Fleischer rings?",
            "What is ATP7B?"
        ]

        for q in simple_queries:
            subqueries = decompose_clinical_query(q)
            self.assertEqual(len(subqueries), 1, f"Simple query '{q}' should not be decomposed")

            candidates, _, _ = self.system._retrieve_hybrid_candidates(q, subqueries)
            self.assertGreater(len(candidates), 0)

            _, coverage = self.system.check_retrieval_coverage(q, subqueries, candidates)

            self.assertTrue(
                coverage["is_complete"],
                f"Simple query '{q}' was incorrectly marked incomplete! Missing: {coverage.get('facets_missing')}"
            )
            self.assertEqual(coverage["coverage_percentage"], 100.0)
            self.assertEqual(len(coverage["facets_missing"]), 0)
            self.assertIsNone(coverage.get("evidence_gap"))

    # =========================================================================
    # 6. PASSAGE PROVENANCE PRESERVATION
    # =========================================================================
    def test_passage_provenance_preserved_in_retrieval(self):
        """
        Verifies that all candidates retained after coverage checking preserve
        proper source document, page number, and formatted citation header.
        """
        query = "What is the role of ceruloplasmin in diagnosing Wilson's Disease?"
        subqueries = decompose_clinical_query(query)
        candidates, _, _ = self.system._retrieve_hybrid_candidates(query, subqueries)
        updated_candidates, _ = self.system.check_retrieval_coverage(query, subqueries, candidates)

        self.assertGreater(len(updated_candidates), 0)
        for idx in updated_candidates:
            p, header, body, body_lower = self.system._parsed_passages[idx]
            self.assertTrue(header.startswith("[Source:"), f"Invalid passage header: {header}")
            self.assertIn("Page:", header)
            # Verify source is in approved guidelines
            source_match = re.search(r'\[Source:\s*([^|]+?)\s*\|\s*Page:\s*(\d+)\s*\]', header)
            self.assertIsNotNone(source_match)
            src_doc = source_match.group(1).strip()
            self.assertIn(src_doc, self.system.approved_docs)

    # =========================================================================
    # 7. TELEMETRY SCHEMA COMPLETENESS
    # =========================================================================
    def test_telemetry_evidence_coverage_structure(self):
        """
        Verifies that telemetry dictionary includes full structured evidence coverage fields:
        facets_expected, facets_covered, facets_missing, coverage_percentage,
        second_pass_performed, and is_complete.
        """
        query = "How is 24-hour urinary copper measured and interpreted?"
        _, telemetry = self.system.ask_chatbot_with_telemetry(query)

        self.assertIn("evidence_coverage", telemetry)
        cov = telemetry["evidence_coverage"]

        required_keys = [
            "facets_expected",
            "facets_covered",
            "facets_missing",
            "coverage_percentage",
            "second_pass_performed",
            "is_complete"
        ]
        for key in required_keys:
            self.assertIn(key, cov, f"Telemetry evidence_coverage missing required field: {key}")

        self.assertIsInstance(cov["facets_expected"], list)
        self.assertIsInstance(cov["facets_covered"], list)
        self.assertIsInstance(cov["facets_missing"], list)
        self.assertIsInstance(cov["coverage_percentage"], float)
        self.assertIsInstance(cov["second_pass_performed"], bool)
        self.assertIsInstance(cov["is_complete"], bool)


if __name__ == "__main__":
    unittest.main()

