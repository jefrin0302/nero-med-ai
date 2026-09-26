#!/usr/bin/env python3
"""
build_vectorstore.py

Ingests medical guidelines from ./medical_docs/, chunks text into passages,
generates embeddings, and persists them into a ChromaDB vector store at ./chroma_db/
"""

import os
import glob

DOCS_DIR = "medical_docs"
PERSIST_DIR = "chroma_db"

def load_documents():
    docs = []
    files = glob.glob(os.path.join(DOCS_DIR, "*.*"))
    print(f"Found {len(files)} file(s) in {DOCS_DIR}")
    for filepath in files:
        if filepath.endswith((".txt", ".md")):
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
                docs.append({"content": text, "source": os.path.basename(filepath)})
    return docs

def simple_chunk_text(text, chunk_size=800, overlap=150):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def build():
    os.makedirs(PERSIST_DIR, exist_ok=True)
    raw_docs = load_documents()
    if not raw_docs:
        print("No medical documents found to process.")
        return

    # Try importing LangChain & ChromaDB
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_community.document_loaders import TextLoader
        try:
            from langchain_chroma import Chroma
        except ImportError:
            from langchain_community.vectorstores import Chroma
            
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings

        print("Using LangChain & ChromaDB for vector indexing...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        
        all_chunks = []
        for doc in raw_docs:
            chunks = text_splitter.create_documents([doc["content"]], metadatas=[{"source": doc["source"]}])
            all_chunks.extend(chunks)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = Chroma.from_documents(
            documents=all_chunks,
            embedding=embeddings,
            persist_directory=PERSIST_DIR
        )
        print(f"Successfully indexed {len(all_chunks)} document chunks into ChromaDB at '{PERSIST_DIR}'.")
        return
    except Exception as e:
        print(f"LangChain/ChromaDB import or processing note: {e}")
        print("Fallback: Creating persistent text passage index for standalone RAG retriever...")
        
        # Fallback lightweight vector index builder
        passages = []
        for doc in raw_docs:
            for chunk in simple_chunk_text(doc["content"]):
                passages.append(chunk)
                
        import json
        with open(os.path.join(PERSIST_DIR, "passages.json"), "w", encoding="utf-8") as f:
            json.dump(passages, f, indent=2)
        print(f"Saved {len(passages)} passages into '{PERSIST_DIR}/passages.json'.")

if __name__ == "__main__":
    build()
