"""
test_rag_guideline_exclusive.py

Unit and Integration Tests verifying that:
1. Medical answers are retrieved exclusively from the 4 approved PDFs in medical_docs/:
   - AASLD WD Guidelines.pdf
   - EASL-ERN Clinical Practice Guidelines on Wilson's disease.pdf
   - NIH Wilson Disease.pdf
   - clinical practice guidelines in WD.pdf
2. Hardcoded medical-answer handlers (Ceruloplasmin, KF rings, ATP7B, Leipzig score, copper tests, etc.)
   cannot return predefined answers.
3. The separate question bank is completely disabled as a source of medical answers.
4. The passage index (chroma_db/passages.json) contains passages exclusively from the 4 approved PDFs.
5. Unanswerable or out-of-guideline queries cleanly state insufficient evidence rather than hallucinating.
6. Answers contain accurate source and page citations.
"""

import os
import json
import re
import unittest
from unittest.mock import MagicMock, patch

from rag_system import MedicalRAGSystem, APPROVED_PDFS, PERSIST_DIR

class TestRAGGuidelineExclusive(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.assistant = MedicalRAGSystem()
        cls.assistant.gemini_client = None
        cls.expected_pdfs = {
            "AASLD WD Guidelines.pdf",
            "EASL-ERN Clinical Practice Guidelines on Wilson's disease.pdf",
            "NIH Wilson Disease.pdf",
            "clinical practice guidelines in WD.pdf"
        }

    def setUp(self):
        # Isolate offline guideline exclusivity assertions from live network API calls
        self.assistant.gemini_client = None

    def test_passages_json_exclusive_to_approved_pdfs(self):
        """Verifies that 100% of passages in the index originate from the 4 approved PDFs."""
        json_path = os.path.join(PERSIST_DIR, "passages.json")
        self.assertTrue(os.path.exists(json_path), f"Passage index not found at {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            passages = json.load(f)

        self.assertGreater(len(passages), 0, "Passages index is empty")

        for idx, passage in enumerate(passages):
            header = passage.split("\n", 1)[0]
            match = re.search(r'\[Source:\s*([^|\]]+)', header)
            self.assertIsNotNone(
                match,
                f"Passage {idx} missing valid [Source: ...] header: {header}"
            )
            src_file = match.group(1).strip()
            self.assertIn(
                src_file,
                self.expected_pdfs,
                f"Passage {idx} has unapproved source: {src_file}"
            )
            self.assertNotIn("question_bank", src_file.lower())
            self.assertNotIn("wilson_disease_question_bank", passage.lower())

    def test_qa_bank_disabled(self):
        """Verifies the question bank is disabled and search_qa_bank returns None."""
        self.assertEqual(self.assistant.qa_bank, [])
        self.assertIsNone(self.assistant.search_qa_bank("What is Ceruloplasmin?"))
        self.assertIsNone(self.assistant.search_qa_bank("What are Kayser-Fleischer rings?"))
        self.assertIsNone(self.assistant.search_qa_bank("What is ATP7B?"))
        self.assertIsNone(self.assistant.search_qa_bank("Any medical query"))

    def test_hardcoded_handlers_removed_for_core_medical_queries(self):
        """
        Verifies that medical queries for Ceruloplasmin, KF rings, ATP7B,
        Leipzig score, Copper tests, etc., route to retrieval and do NOT
        return hardcoded question-bank strings.
        """
        queries_to_test = [
            "What is Ceruloplasmin?",
            "What are Kayser-Fleischer rings?",
            "What is ATP7B?",
            "What is the Leipzig score?",
            "What are the copper based tests?",
            "What is 24 hour urinary copper?",
            "What is copper?",
            "What diet is recommended in Wilson disease?",
            "What are the side effects of Trientine?"
        ]

        forbidden_markers = [
            "Question Bank",
            "Question Bank Sec.",
            "🧪 Ceruloplasmin Overview (Question Bank",
            "👁️ Kayser-Fleischer (KF) Rings (Question Bank",
            "🧬 ATP7B Genetic Profile in Wilson's Disease",
            "📋 Leipzig Consensus Scoring System for Wilson's Disease (Question Bank",
            "🧪 24-Hour Urinary Copper Test (Question Bank",
            "🧪 Copper Metabolism & Pathology in Wilson's Disease (Question Bank"
        ]

        for query in queries_to_test:
            with self.subTest(query=query):
                response = self.assistant.ask_chatbot(query)
                self.assertIsNotNone(response)
                self.assertGreater(len(response), 20)

                # Ensure no hardcoded question-bank markers exist in the response
                for marker in forbidden_markers:
                    self.assertNotIn(
                        marker,
                        response,
                        f"Hardcoded marker '{marker}' found in response to query '{query}'"
                    )

                # Ensure the response cites at least one of the approved PDFs
                has_approved_citation = any(pdf in response for pdf in self.expected_pdfs)
                self.assertTrue(
                    has_approved_citation,
                    f"Response for '{query}' lacks approved PDF citation:\n{response}"
                )

    def test_unanswerable_query_returns_evidence_fallback(self):
        """Verifies that out-of-guideline queries report insufficient evidence rather than hallucinating."""
        query = "Can Wilson disease cause quantum gravitational anomalies in interstellar plasma?"
        response = self.assistant.ask_chatbot(query)
        self.assertIn(
            "do not contain sufficient evidence",
            response.lower(),
            f"Expected insufficient evidence fallback, got:\n{response}"
        )

    def test_conversational_follow_up_grounded_in_approved_pdfs(self):
        """Verifies multi-turn pronoun follow-up retrieves evidence from approved PDFs."""
        history = [
            {"role": "user", "content": "Tell me about Trientine."},
            {"role": "assistant", "content": "Trientine is an oral chelating agent used in Wilson disease."}
        ]
        follow_up_q = "What are its side effects?"
        response = self.assistant.ask_chatbot(follow_up_q, chat_history=history)

        self.assertNotIn("Question Bank", response)
        self.assertNotIn("• Documented Side Effects (Question Bank", response)

        has_approved_citation = any(pdf in response for pdf in self.expected_pdfs)
        self.assertTrue(
            has_approved_citation,
            f"Follow-up response lacks approved PDF citation:\n{response}"
        )

    def test_empty_medical_docs_triggers_no_docs_msg(self):
        """Verifies that when medical_docs has no valid documents, system returns NO_DOCS_MSG."""
        temp_assistant = MedicalRAGSystem()
        with patch.object(temp_assistant, 'has_medical_documents', return_value=False):
            response = temp_assistant.ask_chatbot("What is Ceruloplasmin?")
            self.assertEqual(
                response,
                "I do not have access to any medical documents and cannot answer."
            )


class TestGeminiRAGIntegration(unittest.TestCase):
    """
    Focused unit tests for Gemini RAG integration using a mocked client:
    - Verifies source/page metadata is included in prompt context.
    - Verifies successful Gemini generation returns formatted response with citations.
    - Verifies Gemini exceptions trigger the extractive synthesis fallback.
    - Verifies empty Gemini response triggers the extractive synthesis fallback.
    """
    def setUp(self):
        self.assistant = MedicalRAGSystem()
        self.mock_client = MagicMock()
        self.assistant.gemini_client = self.mock_client

    def test_prompt_includes_source_page_metadata(self):
        """Verifies that [Source: ... | Page: ...] metadata from retrieved passages is present in the prompt."""
        captured_prompt = None

        def mock_generate_content(model, contents):
            nonlocal captured_prompt
            captured_prompt = contents
            res = MagicMock()
            res.text = "Ceruloplasmin is a copper glycoprotein. Source: AASLD WD Guidelines.pdf, p. 7"
            return res

        self.mock_client.models.generate_content.side_effect = mock_generate_content

        known_passage = (
            "[Source: AASLD WD Guidelines.pdf | Page: 7]\n"
            "Serum ceruloplasmin is a copper-binding glycoprotein with ferroxidase activity. "
            "A serum ceruloplasmin level less than 20 mg/dL provides diagnostic evidence for Wilson disease."
        )

        with patch.object(self.assistant, "_rank_passages_internal", return_value=[known_passage]):
            reply = self.assistant.ask_chatbot("What is Ceruloplasmin?")

        self.assertIsNotNone(captured_prompt)
        self.assertIn("CONTEXT PASSAGES:", captured_prompt)
        self.assertIn("[Source: AASLD WD Guidelines.pdf | Page: 7]", captured_prompt)
        self.assertIn("[Source:", captured_prompt)
        self.assertIn("| Page:", captured_prompt)
        self.assertIn("Serum ceruloplasmin is a copper-binding glycoprotein", captured_prompt)
        self.assertIn("AASLD WD Guidelines.pdf, p. 7", reply)

    def test_gemini_valid_citation_accepted(self):
        """Verifies that a Gemini response with a valid citation matching retrieved passages is accepted."""
        res = MagicMock()
        res.text = "Ceruloplasmin is a major copper carrier protein in the blood.\n\nSource: AASLD WD Guidelines.pdf, p. 7"
        self.mock_client.models.generate_content.return_value = res

        reply = self.assistant.ask_chatbot("What is Ceruloplasmin?")
        self.assertIsNotNone(reply)
        self.assertIn("Source: AASLD WD Guidelines.pdf, p. 7", reply)
        self.assertIn("Ceruloplasmin is a major copper carrier protein", reply)

    def test_gemini_fabricated_page_citation_triggers_fallback(self):
        """Verifies that a fabricated page citation from Gemini triggers the extractive synthesis fallback."""
        res = MagicMock()
        res.text = "Ceruloplasmin is a copper glycoprotein.\n\nSource: AASLD WD Guidelines.pdf, p. 999"
        self.mock_client.models.generate_content.return_value = res

        reply = self.assistant.ask_chatbot("What is Ceruloplasmin?")
        self.assertIsNotNone(reply)
        self.assertNotIn("p. 999", reply)
        has_approved_citation = any(pdf in reply for pdf in APPROVED_PDFS)
        self.assertTrue(has_approved_citation, f"Extractive fallback missing approved citation:\n{reply}")
        self.assertIn("ceruloplasmin", reply.lower())

    def test_gemini_missing_citation_triggers_fallback(self):
        """Verifies that a Gemini response missing citations triggers the extractive synthesis fallback."""
        res = MagicMock()
        res.text = "Ceruloplasmin is a copper-binding glycoprotein that transports copper."
        self.mock_client.models.generate_content.return_value = res

        reply = self.assistant.ask_chatbot("What is Ceruloplasmin?")
        self.assertIsNotNone(reply)
        has_approved_citation = any(pdf in reply for pdf in APPROVED_PDFS)
        self.assertTrue(has_approved_citation, f"Extractive fallback missing approved citation:\n{reply}")
        self.assertIn("ceruloplasmin", reply.lower())

    def test_gemini_exception_triggers_extractive_fallback(self):
        """Verifies that a Gemini API failure cleanly triggers the extractive synthesis fallback."""
        self.mock_client.models.generate_content.side_effect = RuntimeError("503 Service Unavailable")

        reply = self.assistant.ask_chatbot("What is Ceruloplasmin?")
        self.assertIsNotNone(reply)
        has_approved_citation = any(pdf in reply for pdf in APPROVED_PDFS)
        self.assertTrue(has_approved_citation, f"Fallback reply missing approved citation:\n{reply}")
        self.assertIn("ceruloplasmin", reply.lower())

    def test_gemini_empty_response_triggers_extractive_fallback(self):
        """Verifies that an empty Gemini response cleanly triggers the extractive synthesis fallback."""
        res = MagicMock()
        res.text = ""
        self.mock_client.models.generate_content.return_value = res

        reply = self.assistant.ask_chatbot("What is Ceruloplasmin?")
        self.assertIsNotNone(reply)
        has_approved_citation = any(pdf in reply for pdf in APPROVED_PDFS)
        self.assertTrue(has_approved_citation, f"Empty-response fallback missing approved citation:\n{reply}")


if __name__ == "__main__":
    unittest.main()
