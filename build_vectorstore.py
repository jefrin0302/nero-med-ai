#!/usr/bin/env python3
"""
build_vectorstore.py

Enhanced Medical Guidelines Knowledge Base Builder for NeuroMed AI.
Ingests medical guidelines exclusively from approved PDF files in ./medical_docs/:
1. AASLD WD Guidelines.pdf
2. EASL-ERN Clinical Practice Guidelines on Wilson's disease.pdf
3. NIH Wilson Disease.pdf
4. clinical practice guidelines in WD.pdf
+ Any Chief Doctor-approved clinical guidelines discovered dynamically from database.py (e.g. Wd_in_pregnancy.pdf).

Features:
- Cleans PyPDF font encoding artifacts (e.g. \\x15, \\ufffd, irregular range hyphens).
- Preserves clinical sentence boundaries, headings, and numerical thresholds.
- Generates stable, deterministic passage IDs (doc_id) using SHA-256.
- Prevents duplicate passage accumulation across indexing runs.
- Persists clean, structured passages into ./chroma_db/passages.json.
- Integrates with ChromaDB/HuggingFaceEmbeddings when available.
"""

import os
import glob
import json
import re
import hashlib
import pypdf

DOCS_DIR = "medical_docs"
PERSIST_DIR = "chroma_db"

APPROVED_PDFS = [
    "AASLD WD Guidelines.pdf",
    "EASL-ERN Clinical Practice Guidelines on Wilson's disease.pdf",
    "NIH Wilson Disease.pdf",
    "clinical practice guidelines in WD.pdf"
]

def clean_clinical_text(raw_text: str) -> str:
    """
    Cleans PyPDF text extraction artifacts, font encoding flaws,
    and irregular dashes/replacement characters while preserving
    medical terms, numbers, units, and clinical sentence structure.
    """
    if not raw_text:
        return ""
    # Normalize non-breaking spaces and line breaks
    t = raw_text.replace('\xa0', ' ').replace('\x0c', '\n').replace('\r\n', '\n').replace('\r', '\n')

    # Fix negative acknowledge \x15 and PDF font anomalies often extracted between ranges (e.g. 2\x155 -> 2-5)
    t = re.sub(r'(\d+)\s*[\x15\ufffd\u2010\u2011\u2012\u2013\u2014\u2015]\s*(\d+)', r'\1-\2', t)

    # Fix any remaining standalone \x15 or \ufffd
    t = t.replace('\x15', '-').replace('\ufffd', '-')

    # Normalize unicode quotation marks and dashes
    t = t.replace('“', '"').replace('”', '"').replace('’', "'").replace('‘', "'")
    t = t.replace('–', '-').replace('—', ' - ')

    # Remove non-printable control characters except standard whitespace (\n, \t)
    t = re.sub(r'[\x00-\x08\x0b\x0e-\x1f\x7f]', '', t)

    # Normalize multiple horizontal spaces
    t = re.sub(r'[ \t]+', ' ', t)
    return t

def get_all_approved_pdf_paths() -> list:
    """
    Discovers all approved PDF filenames and their absolute or relative paths,
    combining the hardcoded baseline with dynamic approvals from database.py.
    """
    targets = []
    seen_files = set()

    # 1. Base approved PDFs
    for name in APPROVED_PDFS:
        path = os.path.join(DOCS_DIR, name)
        if os.path.exists(path):
            targets.append((name, path))
            seen_files.add(name)

    # 2. Dynamic approved documents from database
    try:
        from database import get_rag_documents
        approved_in_db = get_rag_documents(status="approved")
        for doc in approved_in_db:
            fn = doc.get("filename")
            fpath = doc.get("file_path")
            if fn and fn not in seen_files:
                if fpath and os.path.exists(fpath):
                    targets.append((fn, fpath))
                    seen_files.add(fn)
                elif os.path.exists(os.path.join(DOCS_DIR, fn)):
                    targets.append((fn, os.path.join(DOCS_DIR, fn)))
                    seen_files.add(fn)
                elif os.path.exists(os.path.join(DOCS_DIR, "pending", fn)):
                    targets.append((fn, os.path.join(DOCS_DIR, "pending", fn)))
                    seen_files.add(fn)
    except Exception:
        pass

    # 3. Check for Wd_in_pregnancy.pdf if in medical_docs/
    for alt_dir in [DOCS_DIR, os.path.join(DOCS_DIR, "pending")]:
        alt_path = os.path.join(alt_dir, "Wd_in_pregnancy.pdf")
        if os.path.exists(alt_path) and "Wd_in_pregnancy.pdf" not in seen_files:
            targets.append(("Wd_in_pregnancy.pdf", alt_path))
            seen_files.add("Wd_in_pregnancy.pdf")

    return targets

def build():
    os.makedirs(PERSIST_DIR, exist_ok=True)
    passages = []
    seen_passage_hashes = set()

    # Import shared clinical sentence-aware chunker
    from rag_system import chunk_clinical_text

    approved_targets = get_all_approved_pdf_paths()
    print(f"Building clinical knowledge base index exclusively from {len(approved_targets)} approved guidelines...")

    for pdf_name, pdf_path in approved_targets:
        try:
            reader = pypdf.PdfReader(pdf_path)
        except Exception as e:
            print(f"Warning: Failed to read PDF {pdf_path}: {e}")
            continue

        pages = []
        for page_idx, page in enumerate(reader.pages):
            page_num = page_idx + 1
            raw_text = page.extract_text() or ""
            cleaned_text = clean_clinical_text(raw_text)
            if cleaned_text.strip():
                pages.append({"page": page_num, "text": cleaned_text})

        chunks = chunk_clinical_text(
            text="",
            filename=pdf_name,
            target_chunk_size=750,
            min_chunk_size=80,
            overlap=150,
            page_metadata=pages
        )

        added_for_doc = 0
        for c in chunks:
            full_passage = c["full_passage"]
            # Deduplicate by hashing canonical header and normalized text body
            body = c.get("body", "")
            h_key = hashlib.sha256(f"{pdf_name}:{c['page']}:{body[:80]}".encode('utf-8')).hexdigest()
            if h_key not in seen_passage_hashes:
                seen_passage_hashes.add(h_key)
                passages.append(full_passage)
                added_for_doc += 1

        print(f"  - {pdf_name}: {len(reader.pages)} pages, {added_for_doc} clean clinical passages indexed")

    if not passages:
        print("Error: No passages extracted from approved PDFs.")
        return

    # Persist clean passages.json
    json_path = os.path.join(PERSIST_DIR, "passages.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(passages, f, ensure_ascii=False, indent=2)
    print(f"Successfully saved {len(passages)} unique passages to '{json_path}'.")

    # Optional ChromaDB indexing if langchain_chroma / chromadb installed
    try:
        try:
            from langchain_chroma import Chroma
        except ImportError:
            from langchain_community.vectorstores import Chroma

        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings

        model_name = os.environ.get("RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        from langchain_core.documents import Document
        docs = []
        ids = []
        for p in passages:
            hdr, body = p.split("\n", 1) if "\n" in p else ("", p)
            src_m = re.match(r'\[Source:\s*([^|]+)\|\s*Page:\s*(\d+)\]', hdr)
            meta = {}
            source_val = "Unknown"
            page_val = 1
            if src_m:
                source_val = src_m.group(1).strip()
                page_val = int(src_m.group(2).strip())
                meta = {"source": source_val, "page": page_val}
            doc_id = hashlib.sha256(f"{source_val}:{page_val}:{body[:80]}".encode('utf-8')).hexdigest()[:16]
            docs.append(Document(page_content=body, metadata=meta))
            ids.append(doc_id)

        vectorstore = Chroma.from_documents(
            documents=docs,
            ids=ids,
            embedding=embeddings,
            persist_directory=PERSIST_DIR
        )
        print(f"Successfully indexed {len(docs)} documents into ChromaDB at '{PERSIST_DIR}' using '{model_name}'.")
    except Exception as e:
        print(f"ChromaDB embedding note (passages.json is primary): {e}")

if __name__ == "__main__":
    build()
