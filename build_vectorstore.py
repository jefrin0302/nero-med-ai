#!/usr/bin/env python3
"""
build_vectorstore.py

Ingests medical guidelines exclusively from the 4 approved PDF files in ./medical_docs/:
1. AASLD WD Guidelines.pdf
2. EASL-ERN Clinical Practice Guidelines on Wilson's disease.pdf
3. NIH Wilson Disease.pdf
4. clinical practice guidelines in WD.pdf

Extracts text page-by-page, chunks text into passages, and persists them into
the ChromaDB vector store / passage index at ./chroma_db/
"""

import os
import glob
import json
import re
import pypdf

DOCS_DIR = "medical_docs"
PERSIST_DIR = "chroma_db"

APPROVED_PDFS = [
    "AASLD WD Guidelines.pdf",
    "EASL-ERN Clinical Practice Guidelines on Wilson's disease.pdf",
    "NIH Wilson Disease.pdf",
    "clinical practice guidelines in WD.pdf"
]

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list:
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()
        if len(chunk) >= 50:
            chunks.append(chunk)
        if end >= text_len:
            break
        start += chunk_size - overlap
    return chunks

def build():
    os.makedirs(PERSIST_DIR, exist_ok=True)
    passages = []
    
    # Import shared clinical sentence-aware chunker
    from rag_system import chunk_clinical_text
    
    print(f"Building index exclusively from approved PDFs in '{DOCS_DIR}' using clinical sentence-aware chunker...")
    for pdf_name in APPROVED_PDFS:
        pdf_path = os.path.join(DOCS_DIR, pdf_name)
        if not os.path.exists(pdf_path):
            print(f"Warning: Approved PDF not found: {pdf_path}")
            continue
            
        reader = pypdf.PdfReader(pdf_path)
        pages = []
        for page_idx, page in enumerate(reader.pages):
            page_num = page_idx + 1
            raw_text = page.extract_text() or ""
            if raw_text.strip():
                pages.append({"page": page_num, "text": raw_text})
        
        chunks = chunk_clinical_text(
            text="",
            filename=pdf_name,
            target_chunk_size=750,
            min_chunk_size=80,
            overlap=150,
            page_metadata=pages
        )
        for c in chunks:
            passages.append(c["full_passage"])
        print(f"  - {pdf_name}: {len(reader.pages)} pages, {len(chunks)} clinical passages extracted")

    if not passages:
        print("Error: No passages extracted from approved PDFs.")
        return

    # Persist clean passages.json
    json_path = os.path.join(PERSIST_DIR, "passages.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(passages, f, indent=2)
    print(f"Successfully saved {len(passages)} passages to '{json_path}'.")

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
        for p in passages:
            hdr, body = p.split("\n", 1) if "\n" in p else ("", p)
            src_m = re.match(r'\[Source:\s*([^|]+)\|\s*Page:\s*(\d+)\]', hdr)
            meta = {}
            if src_m:
                meta = {"source": src_m.group(1), "page": int(src_m.group(2))}
            docs.append(Document(page_content=body, metadata=meta))

        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=PERSIST_DIR
        )
        print(f"Successfully indexed {len(docs)} documents into ChromaDB at '{PERSIST_DIR}' using '{model_name}'.")
    except Exception as e:
        print(f"ChromaDB embedding note (passages.json is primary): {e}")

if __name__ == "__main__":
    build()
