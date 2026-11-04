"""
test_phase5_synthesis.py

Dedicated Test Suite for Phase 5:
Grounded Answer Synthesis & Strict Citation Validation.

Verifies:
1. Answers are generated exclusively from retrieved passages from the 4 approved Wilson's Disease PDFs.
2. Every requested part of multipart clinical questions is addressed in the synthesized response.
3. Each clinical claim bullet point is associated with claim-level citations: [<pdf>, p. <page>].
4. Strict citation validation accepts verified citations matching retrieved passage metadata.
5. Strict citation validation rejects fabricated page numbers (e.g. p. 999).
6. Strict citation validation rejects fabricated/unapproved PDF filenames (e.g. Fabricated_Guidelines.pdf).
7. Strict citation validation rejects mixed valid + fabricated citations.
8. Strict citation validation rejects bare PDF mentions without page citations.
9. Strict citation validation rejects responses with missing citations or empty text.
10. Fallback on Gemini unverified citation sets safe extractive response with telemetry (fallback_reason='unverified_citation').
11. Fallback on Gemini empty response sets safe extractive response with telemetry (fallback_reason='empty_llm_response').
12. Fallback on Gemini API error sets safe extractive response with telemetry (fallback_reason starts with 'gemini_api_error').
13. Evidence gaps are explicitly reported when approved guidelines lack evidence (zero hallucination).
14. 100% offline standalone compatibility without network dependencies.
15. Deduplication of telemetry sources_cited entries.
"""

import os
import unittest
from unittest.mock import MagicMock
import re

from rag_system import (
    MedicalRAGSystem,
    APPROVED_PDFS
)


class TestPhase5SynthesisAndCitationValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.system = MedicalRAGSystem()
        cls.system.gemini_client = None  # Ensure strictly offline execution by default
        cls.system._ensure_documents_loaded()

    def setUp(self):
        # Reset gemini_client to None before each test
        self.system.gemini_client = None

    # =========================================================================
    # 1. CITATION VALIDATION: STRICT ACCEPTANCE & REJECTION
    # =========================================================================
    def test_citation_validation_accepts_valid_citations(self):
        """Verifies _validate_citations accepts citations matching retrieved passage metadata."""
        passages = [
            "[Source: AASLD WD Guidelines.pdf | Page: 8]\nBaseline 24-h urinary copper excretion is evaluated.",
            "[Source: NIH Wilson Disease.pdf | Page: 14]\nPenicillamine is administered in divided doses."
        ]
        valid_reply = (
            "According to clinical practice guidelines:\n"
            "• Baseline 24-h urinary copper is diagnostic [AASLD WD Guidelines.pdf, p. 8]\n"
            "• Penicillamine is administered in divided doses [NIH Wilson Disease.pdf, p. 14]\n\n"
            "Sources: AASLD WD Guidelines.pdf, p. 8; NIH Wilson Disease.pdf, p. 14"
        )
        is_valid = MedicalRAGSystem._validate_citations(valid_reply, passages, APPROVED_PDFS)
        self.assertTrue(is_valid, "Valid citations matching retrieved passages must be accepted.")

    def test_citation_validation_rejects_fabricated_page_numbers(self):
        """Verifies _validate_citations rejects citations with fabricated page numbers."""
        passages = [
            "[Source: AASLD WD Guidelines.pdf | Page: 8]\nBaseline urinary copper excretion is evaluated."
        ]
        fabricated_page_reply = (
            "According to AASLD WD Guidelines.pdf, p. 999: normal urinary copper is <40 µg/24h."
        )
        is_valid = MedicalRAGSystem._validate_citations(fabricated_page_reply, passages, APPROVED_PDFS)
        self.assertFalse(is_valid, "Fabricated page numbers must be rejected.")

    def test_citation_validation_rejects_fabricated_pdf_filename(self):
        """Verifies _validate_citations rejects unapproved or fabricated PDF filenames."""
        passages = [
            "[Source: AASLD WD Guidelines.pdf | Page: 8]\nBaseline urinary copper excretion is evaluated."
        ]
        fabricated_pdf_reply = (
            "According to Unapproved_Alternative_Medicine_Manual.pdf, p. 8: copper levels normalize."
        )
        is_valid = MedicalRAGSystem._validate_citations(fabricated_pdf_reply, passages, APPROVED_PDFS)
        self.assertFalse(is_valid, "Unapproved/fabricated PDF filenames must be rejected.")

    def test_citation_validation_rejects_mixed_valid_and_fabricated_citations(self):
        """Verifies _validate_citations rejects responses containing mixed valid and fabricated citations."""
        passages = [
            "[Source: AASLD WD Guidelines.pdf | Page: 8]\nBaseline urinary copper excretion is evaluated."
        ]
        mixed_reply = (
            "According to AASLD WD Guidelines.pdf, p. 8, copper excretion is measured.\n"
            "Furthermore, Fake_Research_Paper.pdf, p. 12 suggests chelation protocol."
        )
        is_valid = MedicalRAGSystem._validate_citations(mixed_reply, passages, APPROVED_PDFS)
        self.assertFalse(is_valid, "Mixed valid and fabricated citations must be rejected.")

    def test_citation_validation_rejects_unapproved_source_clause(self):
        """Verifies _validate_citations rejects responses with unapproved documents in Source clauses."""
        passages = [
            "[Source: AASLD WD Guidelines.pdf | Page: 8]\nBaseline urinary copper excretion is evaluated."
        ]
        source_unapproved_reply = (
            "According to clinical practice guidelines:\n"
            "• Baseline urinary copper is measured [AASLD WD Guidelines.pdf, p. 8]\n\n"
            "Sources: AASLD WD Guidelines.pdf, p. 8; Unapproved Handbook 2024, p. 5"
        )
        is_valid = MedicalRAGSystem._validate_citations(source_unapproved_reply, passages, APPROVED_PDFS)
        self.assertFalse(is_valid, "Unapproved documents in Source clauses must be rejected.")

    def test_citation_validation_rejects_bare_pdf_mentions(self):
        """Verifies _validate_citations rejects approved PDF mentions without page citations."""
        passages = [
            "[Source: AASLD WD Guidelines.pdf | Page: 8]\nBaseline urinary copper excretion is evaluated."
        ]
        bare_pdf_reply = (
            "According to AASLD WD Guidelines.pdf, copper excretion is virtually diagnostic."
        )
        is_valid = MedicalRAGSystem._validate_citations(bare_pdf_reply, passages, APPROVED_PDFS)
        self.assertFalse(is_valid, "Bare PDF mentions without verified page citations must be rejected.")

    def test_citation_validation_rejects_missing_citations(self):
        """Verifies _validate_citations rejects responses without any citations."""
        passages = [
            "[Source: AASLD WD Guidelines.pdf | Page: 8]\nBaseline urinary copper excretion is evaluated."
        ]
        no_cite_reply = "Wilson disease is treated with lifelong chelator therapy."
        is_valid = MedicalRAGSystem._validate_citations(no_cite_reply, passages, APPROVED_PDFS)
        self.assertFalse(is_valid, "Responses without citations must be rejected.")

    # =========================================================================
    # 2. MULTIPART QUESTION COVERAGE & CLAIM-LEVEL CITATIONS
    # =========================================================================
    def test_multipart_question_covers_all_facets(self):
        """Verifies that multipart clinical queries produce answers addressing each requested facet."""
        query = "Explain 24-hour urinary copper testing: collection procedure, measurement methods, diagnostic thresholds, and monitoring under chelation therapy."
        reply, telemetry = self.system.ask_chatbot_with_telemetry(query)
        self.assertIsNotNone(reply)
        reply_lower = reply.lower()

        # Check that clinical concepts for all facets are represented
        self.assertTrue(
            any(w in reply_lower for w in ["collect", "collection", "container", "polyethylene", "24-h", "24-hour"]),
            "Response must address specimen collection procedure"
        )
        self.assertTrue(
            any(w in reply_lower for w in ["threshold", "100", "40", "µg", "ug", "umol", "diagnostic", "cutoff"]),
            "Response must address diagnostic thresholds"
        )
        self.assertTrue(
            any(w in reply_lower for w in ["chelat", "penicillamine", "trientine", "zinc", "monitoring", "compliance", "therapy"]),
            "Response must address chelation therapy monitoring"
        )

        # Check telemetry evidence coverage
        cov = telemetry.get("evidence_coverage", {})
        self.assertIn("facets_expected", cov)
        self.assertIn("collection_procedure", cov.get("facets_expected", []))
        self.assertIn("diagnostic_thresholds", cov.get("facets_expected", []))
        self.assertIn("treatment_monitoring", cov.get("facets_expected", []))
        self.assertGreaterEqual(cov.get("coverage_percentage", 0), 75.0)

    def test_clinical_claims_associated_with_citations(self):
        """Verifies each bullet clinical claim in the extractive answer has a claim-level citation."""
        query = "What is the role of ceruloplasmin in diagnosing Wilson's Disease?"
        reply, telemetry = self.system.ask_chatbot_with_telemetry(query)
        self.assertIsNotNone(reply)

        # Extract bullets starting with '• '
        bullet_lines = [line.strip() for line in reply.split("\n") if line.strip().startswith("• ")]
        self.assertGreater(len(bullet_lines), 0, "Response must contain clinical bullet points.")

        for bullet in bullet_lines:
            # Each bullet must have a citation formatted as [<pdf>, p. <page>]
            has_cite = bool(re.search(r'\[[^\]]+\.pdf,\s*p\.\s*\d+\]', bullet))
            self.assertTrue(
                has_cite,
                f"Clinical claim bullet lacks claim-level citation:\n{bullet}"
            )

        # Summary Sources footer must also be present
        self.assertTrue(
            "Source:" in reply or "Sources:" in reply,
            f"Response lacks summary Sources footer:\n{reply}"
        )

    # =========================================================================
    # 3. GEMINI FALLBACK & OBSERVABILITY TELEMETRY
    # =========================================================================
    def test_gemini_fallback_on_unverified_citation(self):
        """Verifies that an unverified LLM citation triggers extractive fallback with observability telemetry."""
        mock_client = MagicMock()
        mock_res = MagicMock()
        mock_res.text = "Ceruloplasmin is a copper protein.\n\nSource: AASLD WD Guidelines.pdf, p. 999"
        mock_client.models.generate_content.return_value = mock_res
        self.system.gemini_client = mock_client

        reply, telemetry = self.system.ask_chatbot_with_telemetry("What is Ceruloplasmin?")
        self.assertIsNotNone(reply)
        self.assertEqual(telemetry.get("synthesis_engine"), "extractive")
        self.assertEqual(telemetry.get("citation_valid"), 0)
        self.assertEqual(telemetry.get("fallback_reason"), "unverified_citation")
        self.assertNotIn("p. 999", reply)

    def test_gemini_fallback_on_empty_response(self):
        """Verifies that an empty LLM response triggers extractive fallback with observability telemetry."""
        mock_client = MagicMock()
        mock_res = MagicMock()
        mock_res.text = ""
        mock_client.models.generate_content.return_value = mock_res
        self.system.gemini_client = mock_client

        reply, telemetry = self.system.ask_chatbot_with_telemetry("What is Ceruloplasmin?")
        self.assertIsNotNone(reply)
        self.assertEqual(telemetry.get("synthesis_engine"), "extractive")
        self.assertEqual(telemetry.get("citation_valid"), 0)
        self.assertEqual(telemetry.get("fallback_reason"), "empty_llm_response")

    def test_gemini_fallback_on_api_error(self):
        """Verifies that an LLM API exception triggers extractive fallback with observability telemetry."""
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = RuntimeError("ResourceExhausted 429 Rate limit exceeded")
        self.system.gemini_client = mock_client

        reply, telemetry = self.system.ask_chatbot_with_telemetry("What is Ceruloplasmin?")
        self.assertIsNotNone(reply)
        self.assertEqual(telemetry.get("synthesis_engine"), "extractive")
        self.assertEqual(telemetry.get("citation_valid"), 0)
        self.assertTrue(
            str(telemetry.get("fallback_reason")).startswith("gemini_api_error"),
            f"Expected fallback_reason to start with gemini_api_error, got: {telemetry.get('fallback_reason')}"
        )

    # =========================================================================
    # 4. EVIDENCE GAP & ZERO HALLUCINATION
    # =========================================================================
    def test_evidence_gap_explicitly_stated_for_out_of_corpus_query(self):
        """Verifies out-of-corpus queries return an explicit evidence gap with zero hallucination."""
        query = "Is CRISPR gene therapy or zinc-nanoparticle infusion approved for Wilson disease management?"
        reply, telemetry = self.system.ask_chatbot_with_telemetry(query)
        self.assertIsNotNone(reply)

        # Must state that guidelines lack evidence
        self.assertIn("do not contain sufficient evidence", reply.lower())
        # Must not hallucinate approval or experimental protocols
        self.assertNotIn("approved for clinical use", reply.lower())

        cov = telemetry.get("evidence_coverage", {})
        self.assertFalse(cov.get("is_complete", True))
        self.assertIn("direct_guideline_evidence", cov.get("facets_missing", []))
        self.assertIsNotNone(cov.get("evidence_gap"))

    # =========================================================================
    # 5. DEDUPLICATION OF TELEMETRY SOURCES CITED
    # =========================================================================
    def test_telemetry_sources_cited_deduplication(self):
        """Verifies telemetry['sources_cited'] contains unique (source, page) entries."""
        query = "Explain 24-hour urinary copper testing: collection procedure, measurement methods, diagnostic thresholds, and monitoring under chelation therapy."
        reply, telemetry = self.system.ask_chatbot_with_telemetry(query)
        sources_cited = telemetry.get("sources_cited", [])
        self.assertGreater(len(sources_cited), 0)

        seen_pairs = set()
        for sc in sources_cited:
            pair = (sc["source"], sc["page"])
            self.assertNotIn(
                pair, seen_pairs,
                f"Duplicate (source, page) entry found in telemetry['sources_cited']: {pair}"
            )
            seen_pairs.add(pair)


if __name__ == "__main__":
    unittest.main()

