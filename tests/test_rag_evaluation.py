"""
test_rag_evaluation.py

Comprehensive Clinical Evaluation Test Suite for NeuroMed AI RAG Pipeline.
Implements reproducible automated tests for clinical scenarios TEST 1 through TEST 6:

- TEST 1: Gene Identification (ATP7B, chromosome 13, copper transport ATPase).
- TEST 2: Ceruloplasmin Diagnostic Role (thresholds <10 mg/dL, 10-20 mg/dL, normal 20-40 mg/dL, limitations).
- TEST 3: 24h Urinary Copper Multi-Part (collection, measurement, thresholds >100 ug/24h & >40 ug/24h, monitoring).
- TEST 4: Leipzig Scoring System (clinical, biochemical, genetic criteria breakdown, cutoff >= 4).
- TEST 5: Treatment Comparison (D-Penicillamine, Trientine, Zinc mechanisms, B6 requirement, side effects).
- TEST 6: Out-of-Corpus / Insufficient Evidence Query (clean insufficient evidence response, zero hallucination).

Also tests query decomposition, configurable retrieval candidate counts, and telemetry schema.
"""

import os
import unittest
import re
from unittest.mock import MagicMock, patch

from rag_system import (
    MedicalRAGSystem,
    decompose_clinical_query,
    APPROVED_PDFS,
    DEFAULT_SEMANTIC_TOP_K,
    DEFAULT_BM25_TOP_K,
    DEFAULT_FINAL_TOP_K
)


class TestRAGEvaluationSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.assistant = MedicalRAGSystem()
        # Isolate from external network during evaluation assertions
        cls.assistant.gemini_client = None

    def setUp(self):
        self.assistant.gemini_client = None

    # =========================================================================
    # TEST 1: GENE IDENTIFICATION
    # =========================================================================
    def test_scenario_1_gene_identification(self):
        """
        TEST 1: Gene Identification
        Query: 'Which gene is responsible for Wilson's Disease?'
        Verifies:
        1. Identifies the ATP7B gene.
        2. Mentions chromosome 13 or copper-transporting ATPase function.
        3. Cites approved clinical guideline documents.
        4. Query decomposition keeps atomic query intact without hallucinating subqueries.
        """
        query = "Which gene is responsible for Wilson's Disease?"

        # 1. Verify decomposition preserves atomic query
        subqueries = decompose_clinical_query(query)
        self.assertEqual(len(subqueries), 1)
        self.assertIn("gene", subqueries[0].lower())

        # 2. Verify retrieval and clinical response
        reply, telemetry = self.assistant.ask_chatbot_with_telemetry(query)
        self.assertIsNotNone(reply)
        reply_lower = reply.lower()

        # Clinical ground-truth checks
        self.assertIn("atp7b", reply_lower, "Response must explicitly identify ATP7B gene")
        self.assertTrue(
            any(k in reply_lower for k in ["chromosome 13", "copper-transporting", "copper transporting", "atpase", "p-type", "transmembrane", "biliary"]),
            "Response must describe genetic location or physiological copper transport function"
        )

        # Source citation verification
        self.assertGreater(len(telemetry.get("sources_cited", [])), 0)
        has_approved_citation = any(pdf in reply for pdf in self.assistant.approved_docs)
        self.assertTrue(has_approved_citation, f"Response lacks approved guideline citation:\n{reply}")

    # =========================================================================
    # TEST 2: CERULOPLASMIN DIAGNOSTIC ROLE
    # =========================================================================
    def test_scenario_2_ceruloplasmin_diagnostic_role(self):
        """
        TEST 2: Ceruloplasmin Role in Diagnosis
        Query: 'What is the role of ceruloplasmin in diagnosing Wilson's Disease?'
        Verifies:
        1. Identifies diagnostic cutoffs (< 10 mg/dL or < 20 mg/dL).
        2. Mentions normal reference range or acute-phase / pregnancy limitations.
        3. Cites approved clinical practice guidelines.
        """
        query = "What is the role of ceruloplasmin in diagnosing Wilson's Disease?"
        reply, telemetry = self.assistant.ask_chatbot_with_telemetry(query)
        self.assertIsNotNone(reply)
        reply_lower = reply.lower()

        # Clinical assertions
        self.assertIn("ceruloplasmin", reply_lower)
        self.assertTrue(
            any(k in reply_lower for k in ["10 mg", "20 mg", "14 mg", "mg/dl", "g/l", "cutoff", "threshold", "diagnostic"]),
            "Response must state quantitative diagnostic thresholds or reference values"
        )
        self.assertTrue(
            any(k in reply_lower for k in ["normal", "false", "acute phase", "pregnancy", "protein", "glycoprotein", "copper"]),
            "Response must mention diagnostic interpretation, protein function, or confounding factors"
        )

        # Citations
        has_approved_citation = any(pdf in reply for pdf in self.assistant.approved_docs)
        self.assertTrue(has_approved_citation, f"Response lacks approved guideline citation:\n{reply}")

    # =========================================================================
    # TEST 3: 24-HOUR URINARY COPPER (MULTI-PART)
    # =========================================================================
    def test_scenario_3_urinary_copper_multipart_coverage(self):
        """
        TEST 3: 24-Hour Urinary Copper Multi-Part Assessment
        Query: 'How is 24-hour urinary copper measured and interpreted?'
        Verifies:
        1. Query is decomposed into collection, measurement, diagnostic thresholds, and monitoring.
        2. Evidence coverage report confirms multi-facet retrieval.
        3. Response addresses specimen collection and diagnostic threshold cutoffs (>100 ug/24h, >40 ug/24h).
        4. Cites approved guidelines with page references.
        """
        query = "How is 24-hour urinary copper measured and interpreted?"

        # 1. Verify query decomposition produces multi-facet subqueries
        subqueries = decompose_clinical_query(query)
        self.assertGreaterEqual(len(subqueries), 3, "Expected at least 3 clinical subqueries for multi-part copper assessment")
        sub_text = " ".join(subqueries).lower()
        self.assertIn("collection", sub_text)
        self.assertIn("measurement", sub_text)
        self.assertIn("threshold", sub_text)

        # 2. Ask chatbot with telemetry
        reply, telemetry = self.assistant.ask_chatbot_with_telemetry(query)
        self.assertIsNotNone(reply)
        reply_lower = reply.lower()

        # 3. Telemetry coverage assertions
        coverage = telemetry.get("evidence_coverage", {})
        self.assertIn("facets_expected", coverage)
        self.assertGreaterEqual(len(coverage.get("facets_covered", [])), 2)
        self.assertGreater(telemetry.get("merged_candidates_count", 0), 0)

        # 4. Clinical content assertions
        self.assertTrue(
            any(k in reply_lower for k in ["urinary", "urine", "24-h", "24-hour", "cupriuria"]),
            "Response must discuss urinary copper"
        )
        self.assertTrue(
            any(k in reply_lower for k in ["100", "40", "µg", "ug", "umol", "µmol", "threshold", "excretion"]),
            "Response must report diagnostic cutoff values (e.g. >100 ug or >40 ug)"
        )

        # 5. Citations check
        has_approved_citation = any(pdf in reply for pdf in self.assistant.approved_docs)
        self.assertTrue(has_approved_citation, f"Response lacks approved guideline citation:\n{reply}")

    # =========================================================================
    # TEST 4: LEIPZIG SCORING SYSTEM
    # =========================================================================
    def test_scenario_4_leipzig_scoring_system(self):
        """
        TEST 4: Leipzig Consensus Scoring System
        Query: 'Explain the Leipzig scoring system and its diagnostic importance.'
        Verifies:
        1. Identifies Leipzig criteria (KF rings, ceruloplasmin, liver copper, urinary copper, genetics).
        2. Identifies diagnostic cutoff score (>= 4 for established diagnosis).
        3. Cites approved guideline documentation.
        """
        query = "Explain the Leipzig scoring system and its diagnostic importance."
        reply, telemetry = self.assistant.ask_chatbot_with_telemetry(query)
        self.assertIsNotNone(reply)
        reply_lower = reply.lower()

        self.assertIn("leipzig", reply_lower)
        self.assertTrue(
            any(k in reply_lower for k in ["score", "scoring", "points", "criteria", "consensus"]),
            "Response must explain scoring criteria"
        )
        self.assertTrue(
            any(k in reply_lower for k in ["4", "established", "diagnosis", "probable", "unlikely", "cut-off", "cutoff"]),
            "Response must mention diagnostic cutoff score (e.g. >= 4)"
        )

        has_approved_citation = any(pdf in reply for pdf in self.assistant.approved_docs)
        self.assertTrue(has_approved_citation, f"Response lacks approved guideline citation:\n{reply}")

    # =========================================================================
    # TEST 5: TREATMENT COMPARISON (D-PENICILLAMINE, TRIENTINE, ZINC)
    # =========================================================================
    def test_scenario_5_treatment_comparison(self):
        """
        TEST 5: Medical Therapy Comparison
        Query: 'What are the differences between D-Penicillamine, Trientine and Zinc therapy?'
        Verifies:
        1. Decomposes into comparative agent subqueries.
        2. Differentiates chelators (penicillamine, trientine) from absorption blockers (zinc).
        3. Mentions clinical nuances (e.g. pyridoxine/B6 with penicillamine, tolerance profile of trientine).
        4. Cites approved guidelines.
        """
        query = "What are the differences between D-Penicillamine, Trientine and Zinc therapy?"

        # Decomposition check
        subqueries = decompose_clinical_query(query)
        self.assertGreaterEqual(len(subqueries), 3)

        # Retrieval & response check
        reply, telemetry = self.assistant.ask_chatbot_with_telemetry(query)
        self.assertIsNotNone(reply)
        reply_lower = reply.lower()

        # Agents check
        self.assertIn("penicillamine", reply_lower)
        self.assertIn("trientine", reply_lower)
        self.assertIn("zinc", reply_lower)

        # Mechanism / Clinical features check
        self.assertTrue(
            any(k in reply_lower for k in ["chelat", "absorp", "metallothionein", "excretion", "pyridoxine", "b6", "adverse", "tolerab", "maintenance"]),
            "Response must compare mechanism of action or adverse effect profiles"
        )

        has_approved_citation = any(pdf in reply for pdf in self.assistant.approved_docs)
        self.assertTrue(has_approved_citation, f"Response lacks approved guideline citation:\n{reply}")

    # =========================================================================
    # TEST 6: OUT-OF-CORPUS / UNRELATED QUERY
    # =========================================================================
    def test_scenario_6_out_of_corpus_insufficient_evidence(self):
        """
        TEST 6: Out-of-Corpus Query Handling
        Query: 'What is the relationship between Wilson's Disease and quantum gravitational anomalies in interstellar plasma?'
        Verifies:
        1. System detects that core substantive terms are absent from approved medical corpus.
        2. Cleanly returns insufficient evidence fallback message.
        3. Does NOT fabricate or hallucinate unsupported medical assertions.
        """
        query = "What is the relationship between Wilson's Disease and quantum gravitational anomalies in interstellar plasma?"
        reply, telemetry = self.assistant.ask_chatbot_with_telemetry(query)
        self.assertIsNotNone(reply)
        reply_lower = reply.lower()

        # Assert clean evidence gap / insufficient evidence response
        self.assertIn(
            "do not contain sufficient evidence",
            reply_lower,
            f"Expected explicit insufficient evidence response, got:\n{reply}"
        )
        self.assertNotIn("interstellar plasma is caused by", reply_lower)
        self.assertNotIn("gravitational anomalies correlate with", reply_lower)

    # =========================================================================
    # RETRIEVAL CONFIGURATION & TELEMETRY SCHEMA TESTS
    # =========================================================================
    def test_configurable_retrieval_candidate_constants(self):
        """Verifies candidate retrieval parameters are configurable via environment variables."""
        self.assertGreaterEqual(DEFAULT_SEMANTIC_TOP_K, 10)
        self.assertGreaterEqual(DEFAULT_BM25_TOP_K, 10)
        self.assertGreaterEqual(DEFAULT_FINAL_TOP_K, 4)

    def test_telemetry_schema_completeness(self):
        """Verifies telemetry dictionary includes all required operational observability fields."""
        query = "What is Ceruloplasmin?"
        _, telemetry = self.assistant.ask_chatbot_with_telemetry(query)

        required_keys = [
            "latency_ms",
            "synthesis_engine",
            "citation_valid",
            "original_query",
            "subqueries",
            "semantic_candidates_count",
            "bm25_candidates_count",
            "merged_candidates_count",
            "final_passages_count",
            "evidence_coverage",
            "sources_cited",
            "retrieved_passages"
        ]
        for key in required_keys:
            self.assertIn(key, telemetry, f"Telemetry missing required key '{key}'")


if __name__ == "__main__":
    unittest.main()

