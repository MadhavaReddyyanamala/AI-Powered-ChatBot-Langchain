import os
from pathlib import Path
from typing import List, Dict
import numpy as np
import faiss
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable

from src.hf_embeddings import HFEmbeddings

class Retriever:
    def __init__(self, index, documents: List[Dict], embedder: HFEmbeddings):
        self.index = index
        self.documents = documents
        self.embedder = embedder
   
    @traceable(name="FAISS Retrieval", run_type="retriever")
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        qvec = self.embedder.embed_query(query)
        distances, indices = self.index.search(qvec, top_k)
        results = []
        for idx in indices[0]:
            if idx == -1:   
                continue
            results.append(self.documents[idx])
        return results

@traceable(name="PDF Extraction", run_type="tool")
def extract_pdf_pages(pdf_path: Path) -> List[Dict]:
    reader = PdfReader(str(pdf_path))
    pages = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({
                "source": pdf_path.name,
                "page": page_num,
                "text": text
            })
    return pages


@traceable(name="Build Retriever", run_type="chain")
def get_retriever(data_dir: str) -> Retriever:
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"{data_dir} folder not found.")

    pdf_files = list(data_path.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError("No PDF files found in the data folder.")

    raw_pages = []
    for pdf in pdf_files:
        raw_pages.extend(extract_pdf_pages(pdf))

    if not raw_pages:
        raise ValueError("No extractable text found. Use text-based PDFs, not scanned-image PDFs.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120)
    chunks = []
    for page in raw_pages:
        split_texts = splitter.split_text(page["text"])
        for chunk in split_texts:
            chunks.append({
                "source": page["source"],
                "page": page["page"],
                "text": chunk
            })

    embedder = HFEmbeddings()
    embeddings = embedder.embed_documents([c["text"] for c in chunks])
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(np.asarray(embeddings, dtype="float32"))

    return Retriever(index=index, documents=chunks, embedder=embedder)