# Clinical RAG Engineering & Governance Rules

## 1. Chief Doctor (Admin) Authorization Gatekeeper for RAG Documents
- Any medical document submitted through clinical portals must initialize with `review_status = 'pending_approval'`.
- Quarantined documents must reside in a separate directory (`medical_docs/pending/`) and MUST NOT be indexed into ChromaDB or `passages.json`.
- The RAG engine (`MedicalRAGSystem`) must strictly restrict retrieval and citation validation to verified, approved documents (`review_status = 'approved'`).
- Only authenticated users with role `admin` (Chief Doctor) may approve or reject submitted documents.
- Upon approval, the document is moved to `medical_docs/`, parsed page-by-page, chunked using the pure shared `chunk_clinical_text()` engine with source/page metadata headers, and dynamically appended to the active in-memory and persisted passage store.
- Rejection must capture the Chief Doctor's reasoning and log a security audit event (`RAG_DOC_REJECTED`).

## 2. Shared Clinical Chunking Engine Invariant
- Both production document ingestion (`ingest_document()`) and interactive visualizers/simulators (`simulate_chunk_division()`) MUST invoke the exact same pure shared function: `chunk_clinical_text(text, filename, target_chunk_size, min_chunk_size, overlap, page_metadata)`.
- Medical abbreviations and dosages (such as `mg.`, `kg.`, `i.e.`, `e.g.`, `vs.`, `tab.`) must be shielded from raw period splitting to preserve critical clinical dosage units.
- Trailing text residues shorter than `min_chunk_size` must be merged into the previous chunk rather than dropped, guaranteeing that no clinical sentences are lost.
- All character start and end offsets must follow Python's standard exclusive indexing convention against a documented reference string.

## 3. Multi-Session Chat Persistence & Conversation Isolation
- Chat conversations must be associated with persistent session records (`chat_sessions`) linked to the authenticated user ID.
- Every user turn and bot reply must be stored in `chat_messages` with timestamp, synthesis engine, and verified citations.
- In-memory client arrays (`chatHistory`) serve only as a display cache; the database is the authoritative source of truth.
- Switching between chat sessions must strictly isolate conversation history to prevent context bleed between distinct patient cases or clinical questions.

## 4. Test State Isolation for Dynamic Knowledge Stores
- Any unit or integration test that exercises document approval, chunk registration, or modifications to `passages.json` must snapshot the original file and in-memory lists in `setUpClass` and restore them in `tearDownClass`.
- Tests must clean up all temporary test files in `medical_docs/` and `medical_docs/pending/` upon completion.
- Vector stores and passage files must remain 100% pure to approved clinical guidelines in production.

## 5. Mock-Aware Telemetry Wrappers
- When enriching core RAG methods with telemetry or structured metrics (e.g. `ask_chatbot_with_telemetry`), wrappers must detect if the underlying method has been patched or mocked by test suites (e.g., `hasattr(fn, "side_effect")` or `hasattr(fn, "_mock_name")`).
- If mocked, the wrapper must route through the mock to ensure test assertions, exception testing, and side-effects behave consistently.
