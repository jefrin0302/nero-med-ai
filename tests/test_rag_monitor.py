"""
test_rag_monitor.py

Comprehensive Automated Test Suite for Clinical RAG Observability, Document Governance,
and Multi-Session Chat Persistence.
Validates:
1. Clinician document submission into quarantined pending state (status='pending_approval').
2. Zero access by RAG chatbot to unapproved / quarantined documents.
3. RBAC authorization gatekeeper: Clinicians cannot approve/reject (HTTP 403).
4. Chief Doctor (Admin) approval workflow: Promotes document, chunks into page-tagged passages,
   dynamically registers into in-memory RAG index without server restart, and updates DB.
5. Chief Doctor rejection workflow with review notes.
6. Persistent multi-session chat: creation, retrieval, turn-by-turn history, and user isolation.
7. Real-time RAG query telemetry logging in rag_query_logs table with KPI metrics summary.
8. /rag/monitor dashboard and /api/rag/test_retrieval diagnostic endpoint.
"""

import os
import io
import time
import json
import shutil
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from app import app, rag_assistant
from database import (
    init_db,
    create_user,
    get_rag_documents,
    get_rag_document_by_id,
    get_user_chat_sessions,
    get_chat_session_messages,
    get_rag_metrics_summary,
    get_recent_rag_queries,
    log_rag_query
)
from rag_system import (
    MedicalRAGSystem,
    ingest_document,
    PERSIST_DIR,
    APPROVED_PDFS,
    detect_clinical_terms,
    chunk_clinical_text,
    simulate_chunk_division,
    parse_passage_header
)


class TestRAGMonitorAndGovernance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        cls._passages_json_path = os.path.join(PERSIST_DIR, "passages.json")
        cls._orig_json_bytes = None
        if os.path.exists(cls._passages_json_path):
            with open(cls._passages_json_path, "rb") as f:
                cls._orig_json_bytes = f.read()
        cls._orig_passages = list(rag_assistant.passages) if hasattr(rag_assistant, "passages") else []
        cls._orig_approved_docs = list(rag_assistant.approved_docs) if hasattr(rag_assistant, "approved_docs") else []

    @classmethod
    def tearDownClass(cls):
        if cls._orig_json_bytes is not None:
            with open(cls._passages_json_path, "wb") as f:
                f.write(cls._orig_json_bytes)
        rag_assistant.passages = list(cls._orig_passages)
        rag_assistant.approved_docs = list(cls._orig_approved_docs)
        if hasattr(rag_assistant, "_load_clinical_passages"):
            rag_assistant._load_clinical_passages()
        # Clean up any test files in medical_docs or medical_docs/pending
        for test_file in ["Pediatric_Trientine_Protocol.txt", "Test_Protocol.txt", "Hospital_Protocol.txt", "Unique_Protocol.txt", "Unverified_Protocol.txt"]:
            prod_path = os.path.join("medical_docs", test_file)
            if os.path.exists(prod_path):
                try:
                    os.remove(prod_path)
                except Exception:
                    pass
        pending_dir = os.path.join("medical_docs", "pending")
        if os.path.exists(pending_dir):
            for f in os.listdir(pending_dir):
                if f.endswith(".txt"):
                    try:
                        os.remove(os.path.join(pending_dir, f))
                    except Exception:
                        pass

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_clinical.db")
        init_db(self.db_path)
        app.config['DB_PATH'] = self.db_path

        # Create test users: clinician and chief doctor (admin)
        self.clinician = create_user("dr_smith", "ClinicianPass123!", "Dr. John Smith", "clinician", db_path=self.db_path)
        self.clinician2 = create_user("dr_jones", "ClinicianPass123!", "Dr. Amy Jones", "clinician", db_path=self.db_path)
        self.chief_doctor = create_user("chief_admin", "AdminPass123!", "Chief Doctor Brown", "admin", db_path=self.db_path)

        self.client = app.test_client()

    def tearDown(self):
        import gc
        gc.collect()
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass
        # Clean up test files in medical_docs/pending
        pending_dir = os.path.join("medical_docs", "pending")
        if os.path.exists(pending_dir):
            for f in os.listdir(pending_dir):
                if f.endswith(".txt"):
                    try:
                        os.remove(os.path.join(pending_dir, f))
                    except Exception:
                        pass

    def _login_as(self, user):
        with self.client.session_transaction() as sess:
            sess["user_id"] = user["user_id"]
            sess["username"] = user["username"]
            sess["role"] = user["role"]
            sess["full_name"] = user["full_name"]
            sess["last_active"] = time.time()

    def test_document_submission_quarantined_pending_approval(self):
        """Verifies that uploaded documents enter 'pending_approval' quarantine state."""
        self._login_as(self.clinician)

        sample_content = b"Clinical Protocol for Wilson Disease Chelation Monitoring.\n\nPage 1 text on dosage."
        data = {
            "title": "Hospital Chelation Protocol",
            "document": (io.BytesIO(sample_content), "Hospital_Protocol.txt")
        }

        resp = self.client.post("/api/rag/upload_document", data=data, content_type="multipart/form-data")
        self.assertEqual(resp.status_code, 201)
        res_json = resp.get_json()
        self.assertTrue(res_json.get("success"))
        self.assertEqual(res_json.get("review_status"), "pending_approval")
        doc_id = res_json.get("doc_id")

        # Verify record in database
        doc = get_rag_document_by_id(doc_id, db_path=self.db_path)
        self.assertIsNotNone(doc)
        self.assertEqual(doc["review_status"], "pending_approval")
        self.assertEqual(doc["uploaded_by_name"], "dr_smith")
        self.assertTrue(os.path.exists(doc["file_path"]))

        # Cleanup pending file
        if os.path.exists(doc["file_path"]):
            try:
                os.remove(doc["file_path"])
            except Exception:
                pass

    def test_duplicate_document_submission_rejected(self):
        """Verifies duplicate documents with identical hash or filename are rejected with HTTP 409."""
        self._login_as(self.clinician)
        sample_content = b"Unique clinical content for Wilson Disease."
        data = {
            "title": "Unique Protocol",
            "document": (io.BytesIO(sample_content), "Unique_Protocol.txt")
        }

        resp1 = self.client.post("/api/rag/upload_document", data=data, content_type="multipart/form-data")
        self.assertEqual(resp1.status_code, 201)

        # Re-upload duplicate
        data_dup = {
            "title": "Duplicate Protocol",
            "document": (io.BytesIO(sample_content), "Unique_Protocol.txt")
        }
        resp2 = self.client.post("/api/rag/upload_document", data=data_dup, content_type="multipart/form-data")
        self.assertEqual(resp2.status_code, 409)

    def test_clinician_cannot_approve_document(self):
        """Verifies non-admin clinician gets HTTP 403 when attempting to approve or reject a document."""
        self._login_as(self.clinician)
        sample_content = b"Protocol text."
        data = {
            "title": "Test Protocol",
            "document": (io.BytesIO(sample_content), "Test_Protocol.txt")
        }
        resp = self.client.post("/api/rag/upload_document", data=data, content_type="multipart/form-data")
        doc_id = resp.get_json()["doc_id"]

        # Clinician attempts approval -> Forbidden
        approve_resp = self.client.post(f"/api/rag/document/{doc_id}/approve", json={"notes": "Clinician self-approve"})
        self.assertEqual(approve_resp.status_code, 403)

        # Clinician attempts rejection -> Forbidden
        reject_resp = self.client.post(f"/api/rag/document/{doc_id}/reject", json={"notes": "Clinician reject"})
        self.assertEqual(reject_resp.status_code, 403)

    def test_chief_doctor_approval_and_dynamic_rag_registration(self):
        """Verifies Chief Doctor can approve document, chunk it, and dynamically register into active RAG."""
        # 1. Clinician submits document
        self._login_as(self.clinician)
        sample_content = (
            "Hospital Clinical Protocol on Wilson Disease Pediatric Chelation.\n\n"
            "Pediatric patients with Wilson disease should initiate Trientine at 20 mg/kg/day divided in 2-3 doses.\n\n"
            "Routine monitoring of 24h urinary copper must be conducted every 3 months during the first year of chelation."
        ).encode("utf-8")
        data = {
            "title": "Pediatric Trientine Protocol",
            "document": (io.BytesIO(sample_content), "Pediatric_Trientine_Protocol.txt")
        }
        sub_resp = self.client.post("/api/rag/upload_document", data=data, content_type="multipart/form-data")
        doc_id = sub_resp.get_json()["doc_id"]

        # 2. Chief Doctor approves document
        self._login_as(self.chief_doctor)
        appr_resp = self.client.post(f"/api/rag/document/{doc_id}/approve", json={"notes": "Approved for clinical use by Chief Doctor Brown"})
        self.assertEqual(appr_resp.status_code, 200)
        appr_json = appr_resp.get_json()
        self.assertTrue(appr_json.get("success"))
        self.assertGreater(appr_json.get("passages_count"), 0)

        # 3. Verify status updated in database
        doc = get_rag_document_by_id(doc_id, db_path=self.db_path)
        self.assertEqual(doc["review_status"], "approved")
        self.assertEqual(doc["reviewed_by_name"], "chief_admin")
        self.assertGreater(doc["passages_count"], 0)

        # 4. Verify RAG assistant in app has document in approved_docs
        from app import rag_assistant
        self.assertIn("Pediatric_Trientine_Protocol.txt", rag_assistant.approved_docs)

        # Clean up created files in medical_docs
        dest_file = os.path.join("medical_docs", "Pediatric_Trientine_Protocol.txt")
        if os.path.exists(dest_file):
            try:
                os.remove(dest_file)
            except Exception:
                pass

    def test_chief_doctor_rejection_workflow(self):
        """Verifies Chief Doctor can reject document with review notes."""
        self._login_as(self.clinician)
        sample_content = b"Unverified protocol."
        data = {
            "title": "Unverified Protocol",
            "document": (io.BytesIO(sample_content), "Unverified_Protocol.txt")
        }
        sub_resp = self.client.post("/api/rag/upload_document", data=data, content_type="multipart/form-data")
        doc_id = sub_resp.get_json()["doc_id"]

        # Chief Doctor rejects
        self._login_as(self.chief_doctor)
        rej_resp = self.client.post(f"/api/rag/document/{doc_id}/reject", json={"notes": "Insufficient clinical evidence."})
        self.assertEqual(rej_resp.status_code, 200)

        doc = get_rag_document_by_id(doc_id, db_path=self.db_path)
        self.assertEqual(doc["review_status"], "rejected")
        self.assertEqual(doc["review_notes"], "Insufficient clinical evidence.")

    def test_multi_session_chat_lifecycle_and_isolation(self):
        """Verifies chat session creation, message persistence, turn retrieval, and user isolation."""
        self._login_as(self.clinician)

        # 1. Create a new session
        new_sess_resp = self.client.post("/api/chat/new_session", json={"title": "Pediatric Consultation"})
        self.assertEqual(new_sess_resp.status_code, 200)
        session_id = new_sess_resp.get_json()["session_id"]
        self.assertIsNotNone(session_id)

        # 2. Send turn via /api/chat with session_id
        chat_resp = self.client.post("/api/chat", json={
            "session_id": session_id,
            "message": "What is the Leipzig score threshold for Wilson disease?"
        })
        self.assertEqual(chat_resp.status_code, 200)
        chat_data = chat_resp.get_json()
        self.assertIn("response", chat_data)
        self.assertEqual(chat_data.get("session_id"), session_id)

        # 3. Retrieve session messages from API
        sess_details_resp = self.client.get(f"/api/chat/session/{session_id}")
        self.assertEqual(sess_details_resp.status_code, 200)
        details = sess_details_resp.get_json()
        messages = details.get("messages", [])
        self.assertGreaterEqual(len(messages), 2)  # User + Assistant
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[1]["role"], "assistant")

        # 4. Verify telemetry recorded in rag_query_logs
        recent_queries = get_recent_rag_queries(limit=10, db_path=self.db_path)
        self.assertGreaterEqual(len(recent_queries), 1)
        self.assertEqual(recent_queries[0]["session_id"], session_id)
        self.assertIn("Leipzig", recent_queries[0]["query_text"])

        # 5. User isolation: Dr. Jones (clinician2) cannot access Dr. Smith's session
        self._login_as(self.clinician2)
        unauth_resp = self.client.get(f"/api/chat/session/{session_id}")
        self.assertEqual(unauth_resp.status_code, 403)

    def test_rag_metrics_summary_calculation(self):
        """Verifies get_rag_metrics_summary calculates pass rate, fallback rate, and averages correctly."""
        # Log 2 successful queries and 1 fallback query
        log_rag_query("query 1", "answer 1", synthesis_engine="gemini", citation_valid=1, latency_ms=120.0, db_path=self.db_path)
        log_rag_query("query 2", "answer 2", synthesis_engine="gemini", citation_valid=1, latency_ms=140.0, db_path=self.db_path)
        log_rag_query("query 3", "answer 3", synthesis_engine="extractive", citation_valid=0, fallback_reason="unverified", latency_ms=80.0, db_path=self.db_path)

        summary = get_rag_metrics_summary(db_path=self.db_path)
        self.assertEqual(summary["total_queries"], 3)
        self.assertEqual(summary["valid_citations"], 2)
        self.assertEqual(summary["fallback_count"], 1)
        self.assertAlmostEqual(summary["pass_rate_pct"], 66.7, places=1)
        self.assertAlmostEqual(summary["fallback_rate_pct"], 33.3, places=1)

    def test_rag_monitor_dashboard_route_and_diagnostic(self):
        """Verifies GET /rag/monitor and POST /api/rag/test_retrieval."""
        self._login_as(self.clinician)

        # GET /rag/monitor
        mon_resp = self.client.get("/rag/monitor")
        self.assertEqual(mon_resp.status_code, 200)
        self.assertIn(b"Clinical RAG Observability", mon_resp.data)
        self.assertIn(b"Document Governance", mon_resp.data)

        # POST /api/rag/test_retrieval
        diag_resp = self.client.post("/api/rag/test_retrieval", json={"query": "Trientine dosing in pediatric Wilson disease"})
        self.assertEqual(diag_resp.status_code, 200)
        diag_data = diag_resp.get_json()
        self.assertTrue(diag_data.get("success"))
        self.assertIsInstance(diag_data.get("passages"), list)

    def test_get_chunk_corpus_global_statistics(self):
        """1. Verifies global corpus statistics calculate totals, averages, and distribution dynamically."""
        corpus = rag_assistant.get_chunk_corpus(offset=0, limit=5)
        self.assertTrue(corpus.get("success"))

        stats = corpus.get("global_statistics") or corpus.get("statistics", {}).get("global")
        self.assertIsNotNone(stats)
        self.assertEqual(stats["total_chunks"], len(rag_assistant.passages))
        self.assertGreater(stats["total_chunks"], 0)
        self.assertGreater(stats["avg_char_length"], 0)
        self.assertLessEqual(stats["min_char_length"], stats["max_char_length"])
        self.assertGreaterEqual(stats["active_documents_count"], 1)

        documents = corpus.get("documents", [])
        self.assertGreaterEqual(len(documents), 1)
        doc_names = [d["document"] for d in documents]
        self.assertTrue(any("AASLD" in d for d in doc_names))

    def test_get_chunk_corpus_filtering_and_pagination(self):
        """2. Verifies case-insensitive document filtering, page filtering, text search, and pagination bounds."""
        corpus_page1 = rag_assistant.get_chunk_corpus(offset=0, limit=5)
        self.assertEqual(len(corpus_page1["chunks"]), 5)
        self.assertTrue(corpus_page1["pagination"]["has_next"])
        self.assertFalse(corpus_page1["pagination"]["has_previous"])

        # Document filter
        first_doc = corpus_page1["documents"][0]["document"]
        filtered_doc = rag_assistant.get_chunk_corpus(doc_filter=first_doc, offset=0, limit=10)
        self.assertTrue(filtered_doc["success"])
        for c in filtered_doc["chunks"]:
            self.assertIn(first_doc.lower(), c["source"].lower())

        # Page filter
        filtered_page = rag_assistant.get_chunk_corpus(page_filter=1, offset=0, limit=10)
        self.assertTrue(filtered_page["success"])
        for c in filtered_page["chunks"]:
            self.assertEqual(c["page"], 1)

        # Keyword text search
        filtered_search = rag_assistant.get_chunk_corpus(search_query="copper", offset=0, limit=10)
        self.assertTrue(filtered_search["success"])
        for c in filtered_search["chunks"]:
            self.assertTrue("copper" in c["body"].lower() or "copper" in c["source"].lower())

    def test_detect_clinical_terms(self):
        """3. Verifies clinical entity detection across biomarkers, genes, treatments, signs, and scoring systems."""
        sample_text = (
            "Patients with homozygous ATP7B mutations typically present with markedly reduced serum ceruloplasmin "
            "and elevated 24-hour urinary copper. Slit-lamp examination may reveal Kayser-Fleischer rings. "
            "A Leipzig score of 4 or higher confirms the diagnosis. Initial decoppering can be initiated with "
            "Trientine or D-penicillamine, while monitoring ALT, AST, and total bilirubin."
        )
        detected = detect_clinical_terms(sample_text)
        self.assertIsInstance(detected, list)
        self.assertGreaterEqual(len(detected), 5)

        categories = {item["category"].lower() for item in detected}
        self.assertIn("gene", categories)
        self.assertIn("biomarker", categories)
        self.assertTrue("sign" in categories or "clinical sign" in categories)
        self.assertTrue("scoring" in categories or "scoring system" in categories)
        self.assertIn("treatment", categories)

        for term_dict in detected:
            self.assertIn("start_char", term_dict)
            self.assertIn("end_char", term_dict)
            self.assertGreaterEqual(term_dict["start_char"], 0)
            self.assertGreater(term_dict["end_char"], term_dict["start_char"])

    def test_shared_chunk_clinical_text_boundaries_and_offsets(self):
        """4. Verifies sentence preservation, medical abbreviation protection, character offsets, and overlap."""
        sample_doc = (
            "The initial dose of Trientine is 20 mg/kg daily in divided doses (e.g., 2 or 3 doses). "
            "In pediatric patients, do not exceed 1000 mg/day initially. "
            "Maintenance therapy should be titrated based on 24-h urinary free copper. "
            "Regular monitoring of serum transaminases (ALT, AST) is mandatory."
        )
        chunks = chunk_clinical_text(
            text=sample_doc,
            filename="Test_Guideline.pdf",
            target_chunk_size=160,
            min_chunk_size=40,
            overlap=30,
            page_metadata={"page": 5}
        )
        self.assertIsInstance(chunks, list)
        self.assertGreaterEqual(len(chunks), 1)

        for chunk in chunks:
            self.assertEqual(chunk["source"], "Test_Guideline.pdf")
            self.assertEqual(chunk["page"], 5)
            self.assertTrue(chunk["header"].startswith("[Source: Test_Guideline.pdf | Page: 5]"))
            self.assertGreaterEqual(chunk["char_count"], 40)
            self.assertGreaterEqual(chunk["word_count"], 5)
            self.assertGreaterEqual(chunk["estimated_tokens"], 1)

            start = chunk["start_char"]
            end = chunk["end_char"]
            self.assertGreaterEqual(start, 0)
            self.assertGreater(end, start)

    def test_shared_chunk_clinical_text_edge_cases(self):
        """5. Verifies handling of empty input, whitespace, Unicode characters, and punctuation."""
        self.assertEqual(chunk_clinical_text("", "Empty.txt"), [])
        self.assertEqual(chunk_clinical_text("   \n\t   \n\n   ", "Whitespace.txt"), [])

        unicode_text = "Wilson-Krankheit: Kupferspeicherkrankheit mit ATP7B-Genmutationen. Dosis: 20 mg/Tag."
        u_chunks = chunk_clinical_text(unicode_text, "German_WD.txt", target_chunk_size=200, min_chunk_size=20)
        self.assertGreaterEqual(len(u_chunks), 1)
        self.assertIn("Wilson-Krankheit", u_chunks[0]["body"])

        no_overlap = chunk_clinical_text(unicode_text, "NoOverlap.txt", target_chunk_size=50, min_chunk_size=20, overlap=0)
        for c in no_overlap:
            self.assertEqual(c["overlap_prev"], "")

    def test_api_rag_chunks_endpoint(self):
        """6. Verifies GET /api/rag/chunks responds with paginated JSON and enforces clinician RBAC."""
        # Unauthenticated request
        anon_resp = self.client.get("/api/rag/chunks")
        self.assertIn(anon_resp.status_code, [302, 401, 403])

        # Authenticated as clinician
        self._login_as(self.clinician)
        resp = self.client.get("/api/rag/chunks?offset=0&limit=5")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data.get("success"))
        self.assertIsInstance(data.get("chunks"), list)
        self.assertLessEqual(len(data.get("chunks")), 5)
        self.assertIn("pagination", data)
        self.assertIn("documents", data)
        self.assertTrue("statistics" in data or "global_statistics" in data)

    def test_api_rag_simulate_chunking_endpoint(self):
        """7. Verifies POST /api/rag/simulate_chunking returns segments, offsets, and overlaps."""
        self._login_as(self.clinician)
        payload = {
            "text": (
                "Wilson disease is caused by mutations in ATP7B gene. "
                "Diagnosis involves serum ceruloplasmin and 24h urinary copper excretion. "
                "Chelation therapy with Trientine or D-penicillamine should be monitored for neurological symptoms."
            ),
            "filename": "Sandbox_Test.pdf",
            "chunk_size": 250,
            "min_chunk_size": 40,
            "overlap": 50,
            "page_metadata": {"page": 2}
        }
        resp = self.client.post("/api/rag/simulate_chunking", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data.get("success"))
        self.assertGreater(data.get("total_chunks", 0), 0)
        self.assertIsInstance(data.get("chunks"), list)
        first_chunk = data["chunks"][0]
        self.assertIn("start_char", first_chunk)
        self.assertIn("end_char", first_chunk)
        self.assertIn("detected_terms", first_chunk)

    def test_simulator_isolation(self):
        """8. Verifies simulation sandbox never alters production in-memory index or disk stores."""
        self._login_as(self.clinician)
        initial_count = len(rag_assistant.passages)
        initial_docs = list(rag_assistant.approved_docs)

        payload = {
            "text": "Experimental simulated guideline text that must never be added to index.",
            "filename": "Experimental_Guideline.pdf",
            "chunk_size": 300,
            "min_chunk_size": 50,
            "overlap": 50
        }
        resp = self.client.post("/api/rag/simulate_chunking", json=payload)
        self.assertEqual(resp.status_code, 200)

        # Active production index must NOT be mutated
        self.assertEqual(len(rag_assistant.passages), initial_count)
        self.assertEqual(list(rag_assistant.approved_docs), initial_docs)
        self.assertFalse(any("Experimental_Guideline" in p for p in rag_assistant.passages))

    def test_chunking_production_simulator_parity(self):
        """9. Verifies exact algorithmic parity between chunk_clinical_text and simulate_chunk_division."""
        sample_text = (
            "The Leipzig scoring system evaluates ceruloplasmin, Kayser-Fleischer rings, "
            "urinary copper excretion, and ATP7B gene mutations. "
            "A total score of 4 or higher establishes a diagnosis of Wilson disease."
        )
        prod_chunks = chunk_clinical_text(
            text=sample_text,
            filename="Parity_Test.pdf",
            target_chunk_size=200,
            min_chunk_size=40,
            overlap=40
        )
        sim_res = simulate_chunk_division(
            text=sample_text,
            filename="Parity_Test.pdf",
            target_chunk_size=200,
            min_chunk_size=40,
            overlap=40
        )
        self.assertEqual(len(prod_chunks), sim_res["total_chunks"])
        self.assertEqual(len(prod_chunks), len(sim_res["chunks"]))
        for p_c, s_c in zip(prod_chunks, sim_res["chunks"]):
            self.assertEqual(p_c["body"], s_c["body"])
            self.assertEqual(p_c["start_char"], s_c["start_char"])
            self.assertEqual(p_c["end_char"], s_c["end_char"])
            self.assertEqual(p_c["header"], s_c["header"])

    def test_chunking_multi_page_provenance(self):
        """10. Verifies preservation of distinct page provenance across multi-page guideline input."""
        multi_pages = [
            {"page": 1, "text": "This is clinical section 1 located on page one of the clinical consensus document."},
            {"page": 2, "text": "This is clinical section 2 located on page two with dosing and follow-up guidance."}
        ]
        chunks = chunk_clinical_text(
            text="",
            filename="MultiPage_Doc.pdf",
            target_chunk_size=300,
            min_chunk_size=30,
            overlap=0,
            page_metadata=multi_pages
        )
        self.assertGreaterEqual(len(chunks), 2)
        pages_found = [c["page"] for c in chunks]
        self.assertIn(1, pages_found)
        self.assertIn(2, pages_found)
        for c in chunks:
            self.assertEqual(c["source"], "MultiPage_Doc.pdf")
            self.assertIn(f"Page: {c['page']}", c["header"])

    def test_chunking_oversized_sentence(self):
        """11. Verifies oversized sentences exceeding target_chunk_size are safely handled without loops."""
        long_sentence = "Wilson " + "disease clinical manifestation diagnosis and chelation monitoring " * 30 + "end."
        target_size = 200
        chunks = chunk_clinical_text(
            text=long_sentence,
            filename="Oversized.pdf",
            target_chunk_size=target_size,
            min_chunk_size=40,
            overlap=30
        )
        self.assertGreater(len(chunks), 1)
        combined_text = " ".join([c["body"] for c in chunks])
        self.assertIn("Wilson", combined_text)
        self.assertIn("end.", combined_text)

    def test_legacy_passage_without_header(self):
        """12. Verifies parse_passage_header handles legacy passages lacking [Source: ... | Page: ...] safely."""
        legacy_text = "This is a raw legacy guideline paragraph without any metadata header."
        src, pg, body = parse_passage_header(legacy_text)
        self.assertEqual(src, "Unknown")
        self.assertIsNone(pg)
        self.assertEqual(body, legacy_text)

    def test_api_rejects_oversized_and_malformed_input(self):
        """13. Verifies POST /api/rag/simulate_chunking rejects oversized text and invalid parameters with HTTP 400."""
        self._login_as(self.clinician)

        # Empty text
        r_empty = self.client.post("/api/rag/simulate_chunking", json={"text": ""})
        self.assertEqual(r_empty.status_code, 400)

        # Text exceeding 100,000 chars
        oversized_text = "A" * 100001
        r_oversized = self.client.post("/api/rag/simulate_chunking", json={"text": oversized_text})
        self.assertEqual(r_oversized.status_code, 400)

        # chunk_size < 200
        r_small = self.client.post("/api/rag/simulate_chunking", json={"text": "Valid text", "chunk_size": 100})
        self.assertEqual(r_small.status_code, 400)

        # chunk_size > 2000
        r_huge = self.client.post("/api/rag/simulate_chunking", json={"text": "Valid text", "chunk_size": 3000})
        self.assertEqual(r_huge.status_code, 400)

        # overlap >= chunk_size
        r_overlap = self.client.post("/api/rag/simulate_chunking", json={"text": "Valid text", "chunk_size": 300, "overlap": 350})
        self.assertEqual(r_overlap.status_code, 400)

    def test_explorer_excludes_unapproved_documents(self):
        """14. Verifies unapproved (pending and rejected) documents are completely excluded from explorer."""
        self._login_as(self.clinician)

        # 1. Submit a document as clinician -> lands in pending_approval
        data = {
            "title": "Quarantined_Protocol",
            "document": (io.BytesIO(b"Quarantined unapproved content."), "Quarantined_Protocol.txt")
        }
        upload_resp = self.client.post(
            "/api/rag/upload_document",
            data=data,
            content_type="multipart/form-data"
        )
        self.assertIn(upload_resp.status_code, (200, 201))

        # 2. Query corpus explorer via Python service and HTTP endpoint
        corpus = rag_assistant.get_chunk_corpus()
        for doc in corpus["documents"]:
            self.assertNotIn("Quarantined_Protocol", doc["document"])
        for chunk in corpus["chunks"]:
            self.assertNotIn("Quarantined_Protocol", chunk["source"])

        api_resp = self.client.get("/api/rag/chunks")
        self.assertEqual(api_resp.status_code, 200)
        api_data = api_resp.get_json()
        for doc in api_data.get("documents", []):
            self.assertNotIn("Quarantined_Protocol", doc["document"])

    def test_api_rag_chunk_stats_endpoint(self):
        """Verifies GET /api/rag/chunk_stats lightweight metrics."""
        self._login_as(self.clinician)
        resp = self.client.get("/api/rag/chunk_stats")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data.get("success"))
        self.assertIn("global_statistics", data)
        self.assertIn("documents", data)

    def test_rag_monitor_dashboard_rendered_tabs_and_preloads(self):
        """Verifies GET /rag/monitor renders Chunks Explorer and Architecture Pipeline elements."""
        self._login_as(self.clinician)
        resp = self.client.get("/rag/monitor")
        self.assertEqual(resp.status_code, 200)
        html = resp.data.decode("utf-8")

        # Tab buttons
        self.assertIn("Chunks Explorer", html)
        self.assertTrue("How It Works & Divides" in html or "How It Works &amp; Divides" in html)

        # Chunks Explorer Elements
        self.assertIn('id="tab-chunks"', html)
        self.assertIn('id="chunk-doc-filter"', html)
        self.assertIn('id="chunks-cards-grid"', html)

        # Architecture & Simulator Elements
        self.assertIn('id="tab-pipeline"', html)
        self.assertIn('id="stage-card-1"', html)
        self.assertIn('id="sim-input-text"', html)
        self.assertIn('runChunkSimulation()', html)


if __name__ == "__main__":
    unittest.main()

