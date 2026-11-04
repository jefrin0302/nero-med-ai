"""
test_phase4_reranking.py

Dedicated Test Suite for Phase 4:
Coverage-Preserving Reranking & Diagnostic Density Scoring.

Verifies:
1. BM25 canonical token consistency across queries and indexed corpus.
2. Robustness across diverse Leipzig query phrasing variations (including participles: 'scoring', 'diagnosing').
3. High diagnostic density scoring preference (thresholds, criteria, recommendation statements).
4. Near-duplicate overlapping passage suppression.
5. Preservation of all required evidence facets in final context.
6. Citation validity and passage provenance preservation.
7. Post-reranking evidence coverage recheck verification.
8. Second-pass retrieval when facets are initially missing.
9. Strict out-of-corpus fallback safety and zero hallucination.
10. Standalone offline compatibility without network dependencies.
"""

import os
import unittest
import re
from rag_system import (
    MedicalRAGSystem,
    _canonical_token,
    compute_diagnostic_density_score,
    is_near_duplicate_passage,
    decompose_clinical_query,
    APPROVED_PDFS,
    DEFAULT_FINAL_TOP_K
)


class TestPhase4RerankingAndDensity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.system = MedicalRAGSystem()
        cls.system.gemini_client = None  # Ensure strictly offline execution
        cls.system._ensure_documents_loaded()

    def setUp(self):
        self.system.gemini_client = None

    # =========================================================================
    # 1. BM25 CANONICAL TOKEN CONSISTENCY
    # =========================================================================
    def test_bm25_canonical_token_consistency(self):
        """
        Verifies that _canonical_token consistently normalizes inflections
        and protects medical terminology without destructive over-stemming.
        """
        test_pairs = [
            ("scoring", "score"),
            ("scores", "score"),
            ("score", "score"),
            ("diagnosis", "diagnosis"),
            ("diagnosing", "diagnosis"),
            ("diagnostic", "diagnosis"),
            ("diagnosed", "diagnosis"),
            ("cirrhosis", "cirrhosis"),
            ("menkes", "menkes"),
            ("steatosis", "steatosis"),
            ("criteria", "criteria"),
            ("atp7b", "atp7b"),
            ("leipzig", "leipzig"),
            ("penicillamine", "penicillamine"),
            ("trientine", "trientine"),
            ("zinc", "zinc")
        ]
        for raw, expected in test_pairs:
            self.assertEqual(
                _canonical_token(raw), expected,
                f"Token canonicalization failed for '{raw}': got '{_canonical_token(raw)}', expected '{expected}'"
            )

        # Ensure corpus BM25 index contains canonical terms with non-zero document frequencies
        for term in ["score", "diagnosis", "criteria", "leipzig", "atp7b", "cirrhosis"]:
            df = self.system._bm25_df.get(term, 0)
            self.assertGreater(df, 0, f"Expected non-zero BM25 document frequency for canonical term '{term}'")

    # =========================================================================
    # 2. LEIPZIG QUERY PHRASING VARIATIONS
    # =========================================================================
    def test_leipzig_query_phrasing_variations(self):
        """
        Verifies that diverse query phrasings for the Leipzig consensus scoring system
        successfully retrieve guidelines and achieve 100% coverage across all 4 criteria facets.
        """
        phrasings = [
            "Explain the Leipzig scoring system and its diagnostic importance.",
            "What are the criteria and scoring points in the Leipzig scoring system for diagnosing Wilson's Disease?",
            "How is the Leipzig score calculated for Wilson disease diagnosis?"
        ]

        for query in phrasings:
            reply, telemetry = self.system.ask_chatbot_with_telemetry(query)
            self.assertIsNotNone(reply)

            cov = telemetry.get("evidence_coverage", {})
            self.assertTrue(
                cov.get("is_complete"),
                f"Query '{query}' failed complete coverage: missing {cov.get('facets_missing')}"
            )
            self.assertEqual(cov.get("coverage_percentage"), 100.0)
            self.assertEqual(len(cov.get("facets_missing", [])), 0)
            self.assertGreaterEqual(telemetry.get("final_passages_count", 0), 4)

            # Assert clinical criteria presence in reply
            reply_lower = reply.lower()
            self.assertTrue(any(w in reply_lower for w in ["leipzig", "score", "table"]))

    # =========================================================================
    # 3. DIAGNOSTIC INFORMATION DENSITY SCORING
    # =========================================================================
    def test_diagnostic_density_scoring_evaluates_clinical_richness(self):
        """
        Verifies compute_diagnostic_density_score rewards passages containing
        quantitative cutoffs, recommendation statements, and treatment mechanisms.
        """
        high_density_text = (
            "Class I, Level A guidance statement: We recommend that 24-h urinary copper excretion "
            ">100 µg/24h (>1.6 µmol/24h) in symptomatic patients or >40 µg/24h confirms Wilson disease. "
            "Modified Leipzig score >= 4 points establishes diagnosis. Initial de-coppering therapy with "
            "D-penicillamine 1000-1500 mg/day requires 25-50 mg/day pyridoxine supplementation."
        )
        low_density_text = (
            "Wilson disease was studied in several patients over many years. "
            "Further clinical investigations were conducted according to routine practice."
        )

        score_high = compute_diagnostic_density_score(high_density_text, high_density_text.lower())
        score_low = compute_diagnostic_density_score(low_density_text, low_density_text.lower())

        self.assertGreater(score_high["total_density"], score_low["total_density"] + 8.0)
        self.assertGreater(score_high["diagnostic_thresholds"], 3.0)
        self.assertGreater(score_high["guideline_recommendations"], 2.0)
        self.assertGreater(score_high["clinical_criteria"], 2.0)
        self.assertGreater(score_high["treatment_mechanisms"], 1.0)

    # =========================================================================
    # 4. DUPLICATE PASSAGE SUPPRESSION
    # =========================================================================
    def test_near_duplicate_passage_suppression(self):
        """
        Verifies is_near_duplicate_passage detects high token overlap between
        adjacent sliding-window chunks and prevents redundancy.
        """
        base_passage = (
            "Basal urinary copper excretion without chelating agents is almost invariably elevated "
            "above 40 ug or ~0.6 umol/24 hours in most individuals with Wilson disease, and above 100 ug in symptomatic individuals."
        )
        near_dup = (
            "Basal urinary copper excretion without chelating agents is almost invariably elevated "
            "above 40 ug or ~0.6 umol/24 hours in most individuals with Wilson disease, and above 100 ug in symptomatic individuals. "
            "Testing laboratories should be consulted."
        )
        distinct_passage = (
            "Trientine is an alternative copper chelating agent for patients intolerant of D-penicillamine. "
            "It acts by promoting urinary copper excretion but has a different chemical structure and adverse effect profile."
        )

        self.assertTrue(is_near_duplicate_passage(near_dup, [base_passage], threshold=0.65))
        self.assertFalse(is_near_duplicate_passage(distinct_passage, [base_passage], threshold=0.65))

    # =========================================================================
    # 5. PRESERVATION OF ALL REQUIRED EVIDENCE FACETS
    # =========================================================================
    def test_facet_preservation_in_multipart_urinary_copper(self):
        """
        Verifies that final top-k context retains passages covering all 4 required facets
        and does not drop the only passage supporting an evidence facet.
        """
        query = "How is 24-hour urinary copper collected, measured, and interpreted in Wilson's Disease?"
        subqueries = decompose_clinical_query(query)
        candidates, _, _ = self.system._retrieve_hybrid_candidates(query, subqueries)
        candidates, pre_cov = self.system.check_retrieval_coverage(query, subqueries, candidates)

        final_passages, post_cov = self.system._rerank_candidates_with_coverage(
            query, subqueries, candidates, top_n=DEFAULT_FINAL_TOP_K
        )

        self.assertEqual(len(final_passages), DEFAULT_FINAL_TOP_K)
        self.assertTrue(post_cov["is_complete"])
        self.assertEqual(post_cov["coverage_percentage"], 100.0)

        # Inspect passage bodies to verify actual facet coverage in final list
        combined_text = " ".join(final_passages).lower()
        self.assertTrue(any(w in combined_text for w in ["collect", "container", "protocol", "24-h", "24-hour"]))
        self.assertTrue(any(w in combined_text for w in ["measured", "measurement", "assay", "spectrophotometry"]))
        self.assertTrue(any(w in combined_text for w in [">100", "> 100", "100 µg", "100 ug", "40 µg", "40 ug", "threshold"]))
        self.assertTrue(any(w in combined_text for w in ["monitoring", "target", "penicillamine", "trientine", "zinc"]))

    # =========================================================================
    # 6. CITATION & PROVENANCE PRESERVATION
    # =========================================================================
    def test_citation_and_provenance_preservation(self):
        """
        Verifies that all reranked passages maintain valid [Source: <filename> | Page: <page>] headers
        pointing strictly to approved clinical practice guidelines.
        """
        query = "What is the role of ceruloplasmin in diagnosing Wilson's Disease?"
        subqueries = decompose_clinical_query(query)
        candidates, _, _ = self.system._retrieve_hybrid_candidates(query, subqueries)
        final_passages, _ = self.system._rerank_candidates_with_coverage(query, subqueries, candidates)

        self.assertGreater(len(final_passages), 0)
        for p in final_passages:
            header = p.split("\n", 1)[0]
            m = re.match(r'^\[Source:\s*([^|]+?)\s*\|\s*Page:\s*(\d+)\s*\]$', header)
            self.assertIsNotNone(m, f"Passage missing standard header: {header}")
            doc_name = m.group(1).strip()
            self.assertIn(doc_name, self.system.approved_docs)
            self.assertGreater(int(m.group(2).strip()), 0)

    # =========================================================================
    # 7. POST-RERANKING EVIDENCE COVERAGE VERIFICATION
    # =========================================================================
    def test_post_reranking_coverage_telemetry_matches(self):
        """
        Verifies that telemetry['evidence_coverage'] reflects the verified post-reranking coverage.
        """
        query = "What is the difference between D-penicillamine, Trientine, and Zinc in Wilson disease?"
        _, telemetry = self.system.ask_chatbot_with_telemetry(query)

        cov = telemetry.get("evidence_coverage", {})
        self.assertIn("facets_expected", cov)
        self.assertIn("facets_covered", cov)
        self.assertIn("is_complete", cov)
        self.assertTrue(cov["is_complete"])
        self.assertEqual(cov["coverage_percentage"], 100.0)
        self.assertIn("d_penicillamine", cov["facets_covered"])
        self.assertIn("trientine", cov["facets_covered"])
        self.assertIn("zinc_therapy", cov["facets_covered"])

    # =========================================================================
    # 8. SECOND-PASS RETRIEVAL RECOVERS MISSING FACETS
    # =========================================================================
    def test_second_pass_retrieval_expansion(self):
        """
        Verifies that if an initial candidate set has missing facets, second-pass
        retrieval triggers targeted search and recovers coverage.
        """
        query = "What is the difference between D-penicillamine, Trientine, and Zinc in Wilson disease?"
        subqueries = decompose_clinical_query(query)

        # Artificial set with only zinc passage
        initial_set = set()
        for idx, (p, h, b, b_low) in enumerate(self.system._parsed_passages):
            if "zinc" in b_low and "penicillamine" not in b_low and "trientine" not in b_low:
                initial_set.add(idx)
                break

        updated_candidates, cov = self.system.check_retrieval_coverage(query, subqueries, set(initial_set))
        self.assertTrue(cov["second_pass_performed"])
        self.assertGreater(len(updated_candidates), len(initial_set))
        self.assertIn("d_penicillamine", cov["facets_covered"])
        self.assertIn("trientine", cov["facets_covered"])

    # =========================================================================
    # 9. OUT-OF-CORPUS FALLBACK SAFETY
    # =========================================================================
    def test_out_of_corpus_query_safely_falls_back(self):
        """
        Verifies that completely out-of-corpus queries safely trigger the evidence fallback
        without hallucinating cutoffs, dosages, or citations.
        """
        query = "What is the clinical protocol for CAR-T cell immunotherapy in Wilson's disease?"
        reply, telemetry = self.system.ask_chatbot_with_telemetry(query)

        self.assertIn("do not contain sufficient evidence", reply.lower())
        self.assertEqual(telemetry.get("final_passages_count"), 0)
        self.assertEqual(len(telemetry.get("sources_cited", [])), 0)
        cov = telemetry.get("evidence_coverage", {})
        self.assertFalse(cov.get("is_complete"))
        self.assertEqual(cov.get("coverage_percentage"), 0.0)

    # =========================================================================
    # 10. OFFLINE COMPATIBILITY
    # =========================================================================
    def test_standalone_offline_operation(self):
        """
        Verifies that candidate retrieval, BM25 indexing, dense embedding search,
        and diagnostic density reranking operate with zero internet connection.
        """
        self.assertIsNone(self.system.gemini_client)
        query = "Which gene is responsible for Wilson's Disease?"
        reply, telemetry = self.system.ask_chatbot_with_telemetry(query)

        self.assertIsNotNone(reply)
        self.assertEqual(telemetry.get("synthesis_engine"), "extractive")
        self.assertIn("atp7b", reply.lower())
        self.assertGreater(len(telemetry.get("sources_cited", [])), 0)


if __name__ == "__main__":
    unittest.main()
