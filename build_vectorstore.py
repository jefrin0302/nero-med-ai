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
    
    print(f"Building index exclusively from approved PDFs in '{DOCS_DIR}'...")
    for pdf_name in APPROVED_PDFS:
        pdf_path = os.path.join(DOCS_DIR, pdf_name)
        if not os.path.exists(pdf_path):
            print(f"Warning: Approved PDF not found: {pdf_path}")
            continue
            
        reader = pypdf.PdfReader(pdf_path)
        pdf_chunks_count = 0
        for page_idx, page in enumerate(reader.pages):
            page_num = page_idx + 1
            raw_text = page.extract_text() or ""
            # Clean horizontal whitespace
            clean_text = re.sub(r'[ \t]+', ' ', raw_text)
            page_chunks = chunk_text(clean_text)
            for c in page_chunks:
                tagged_passage = f"[Source: {pdf_name} | Page: {page_num}]\n{c}"
                passages.append(tagged_passage)
                pdf_chunks_count += 1
        print(f"  - {pdf_name}: {len(reader.pages)} pages, {pdf_chunks_count} passages extracted")

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

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
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
        print(f"Successfully indexed {len(docs)} documents into ChromaDB at '{PERSIST_DIR}'.")
    except Exception as e:
        print(f"ChromaDB embedding note (passages.json is primary): {e}")

if __name__ == "__main__":
    build()
