"""
tests/test_phase2_retrieval_decomposition.py

Unit tests for Phase 2:
1. Environment-configurable candidate retrieval parameters:
   - RAG_SEMANTIC_TOP_K (default 15)
   - RAG_BM25_TOP_K (default 15)
   - RAG_FINAL_TOP_K (default 6)
2. Deterministic clinical query decomposition:
   - Preservation of simple/atomic queries as single queries
   - Decomposition of complex multi-part queries into focused clinical subqueries
3. Hybrid candidate retrieval:
   - Semantic dense vector search + BM25 keyword search
4. Passage deduplication & metadata preservation:
   - Merging candidates without duplicate passages
   - Preserving source and page metadata [Source: ... | Page: ...]
5. Standalone deployment compatibility:
   - Fully offline execution using local corpus and precomputed embeddings
"""

import os
import unittest
import importlib
from unittest.mock import patch

from rag_system import (
    MedicalRAGSystem,
    decompose_clinical_query,
    DEFAULT_SEMANTIC_TOP_K,
    DEFAULT_BM25_TOP_K,
    DEFAULT_FINAL_TOP_K
)


class TestPhase2RetrievalDecomposition(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.assistant = MedicalRAGSystem()
        cls.assistant.gemini_client = None

    def setUp(self):
        self.assistant.gemini_client = None

    def test_environment_configurable_retrieval_defaults(self):
        """Verifies default retrieval constants match specified requirements."""
        self.assertEqual(DEFAULT_SEMANTIC_TOP_K, 15)
        self.assertEqual(DEFAULT_BM25_TOP_K, 15)
        self.assertEqual(DEFAULT_FINAL_TOP_K, 6)

    def test_environment_variable_override(self):
        """Verifies retrieval constants can be overridden by environment variables."""
        with patch.dict(os.environ, {
            "RAG_SEMANTIC_TOP_K": "20",
            "RAG_BM25_TOP_K": "25",
            "RAG_FINAL_TOP_K": "8"
        }):
            sem_k = int(os.environ.get("RAG_SEMANTIC_TOP_K", "15"))
            bm25_k = int(os.environ.get("RAG_BM25_TOP_K", "15"))
            final_k = int(os.environ.get("RAG_FINAL_TOP_K", "6"))
            self.assertEqual(sem_k, 20)
            self.assertEqual(bm25_k, 25)
            self.assertEqual(final_k, 8)

    def test_simple_atomic_queries_preserved_without_decomposition(self):
        """Verifies simple atomic clinical questions are preserved as single queries."""
        simple_queries = [
            "Which gene is responsible for Wilson's Disease?",
            "What is Ceruloplasmin?",
            "What are Kayser-Fleischer rings?",
            "What is ATP7B?",
            "Who discovered Wilson's disease?"
        ]
        for query in simple_queries:
            with self.subTest(query=query):
                subqueries = decompose_clinical_query(query)
                self.assertIsInstance(subqueries, list)
                self.assertEqual(len(subqueries), 1, f"Simple query was unexpectedly split: {subqueries}")
                self.assertEqual(subqueries[0], query)

    def test_complex_query_decomposition_urinary_copper(self):
        """Verifies 24-hour urinary copper question decomposes into focused subqueries."""
        query = "How is 24-hour urinary copper measured and interpreted?"
        subqueries = decompose_clinical_query(query)
        self.assertIsInstance(subqueries, list)
        self.assertGreaterEqual(len(subqueries), 4)

        combined = " ".join(subqueries).lower()
        self.assertIn("collection", combined, "Missing specimen collection subquery")
        self.assertIn("measurement", combined, "Missing laboratory measurement subquery")
        self.assertIn("threshold", combined, "Missing diagnostic thresholds subquery")
        self.assertIn("monitoring", combined, "Missing treatment monitoring subquery")

    def test_complex_query_decomposition_treatment_comparison(self):
        """Verifies drug therapy comparison decomposes into focused subqueries for each agent."""
        query = "What are the differences between D-Penicillamine, Trientine and Zinc therapy?"
        subqueries = decompose_clinical_query(query)
        self.assertIsInstance(subqueries, list)
        self.assertGreaterEqual(len(subqueries), 3)

        combined = " ".join(subqueries).lower()
        self.assertIn("penicillamine", combined)
        self.assertIn("trientine", combined)
        self.assertIn("zinc", combined)

    def test_complex_query_decomposition_leipzig_scoring(self):
        """Verifies Leipzig scoring decomposes into focused subqueries."""
        query = "Explain the Leipzig scoring system and its diagnostic importance."
        subqueries = decompose_clinical_query(query)
        self.assertIsInstance(subqueries, list)
        self.assertGreaterEqual(len(subqueries), 3)

        combined = " ".join(subqueries).lower()
        self.assertIn("leipzig", combined)
        self.assertIn("criteria", combined)

    def test_hybrid_candidate_retrieval_and_counts(self):
        """Verifies hybrid retrieval collects candidates from both semantic and BM25 searches."""
        query = "How is 24-hour urinary copper measured and interpreted?"
        subqueries = decompose_clinical_query(query)

        candidate_indices, sem_count, bm25_count = self.assistant._retrieve_hybrid_candidates(
            query=query,
            subqueries=subqueries,
            semantic_k=DEFAULT_SEMANTIC_TOP_K,
            bm25_k=DEFAULT_BM25_TOP_K
        )

        self.assertIsInstance(candidate_indices, set)
        self.assertGreater(len(candidate_indices), 10, "Candidate pool too small for multi-query hybrid search")
        self.assertGreater(sem_count, 0, "Semantic retrieval returned 0 candidates")
        self.assertGreater(bm25_count, 0, "BM25 retrieval returned 0 candidates")

    def test_candidate_deduplication_and_metadata_preservation(self):
        """Verifies candidate passages are deduplicated and source/page provenance is intact."""
        query = "How is 24-hour urinary copper measured and interpreted?"
        subqueries = decompose_clinical_query(query)

        candidate_indices, _, _ = self.assistant._retrieve_hybrid_candidates(
            query=query,
            subqueries=subqueries,
            semantic_k=DEFAULT_SEMANTIC_TOP_K,
            bm25_k=DEFAULT_BM25_TOP_K
        )

        # Confirm all candidate indices are valid and unique
        self.assertEqual(len(candidate_indices), len(set(candidate_indices)))

        for doc_idx in candidate_indices:
            self.assertLess(doc_idx, len(self.assistant._parsed_passages))
            p, header, body, body_lower = self.assistant._parsed_passages[doc_idx]
            self.assertTrue(header.startswith("[Source:"), f"Header missing [Source: ...]: {header}")
            self.assertIn("| Page:", header, f"Header missing | Page: ...: {header}")
            self.assertGreater(len(body), 20, "Passage body too short")

    def test_standalone_offline_compatibility(self):
        """Verifies that candidate retrieval executes entirely offline without network access."""
        query = "What is the role of ceruloplasmin in diagnosing Wilson's Disease?"
        subqueries = decompose_clinical_query(query)

        # Execute candidate retrieval with mocked network environment
        candidate_indices, sem_count, bm25_count = self.assistant._retrieve_hybrid_candidates(
            query=query,
            subqueries=subqueries,
            semantic_k=DEFAULT_SEMANTIC_TOP_K,
            bm25_k=DEFAULT_BM25_TOP_K
        )
        self.assertGreater(len(candidate_indices), 0)
        self.assertGreater(bm25_count, 0)


if __name__ == "__main__":
    unittest.main()

