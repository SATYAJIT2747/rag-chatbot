import os
import pickle
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi

class LocalBM25Retriever:
    def __init__(self, chunks: List[Document]):
        self.chunks = chunks
        self.corpus = [c.page_content.lower().split() for c in chunks]
        self.bm25 = BM25Okapi(self.corpus)

    def retrieve(self, query: str, top_k: int = 20) -> List[Tuple[Document, float, int]]:
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for rank, idx in enumerate(sorted_indices, 1):
            results.append((self.chunks[idx], float(scores[idx]), rank))
        return results

def chunk_documents(documents: List[Document], chunk_size: int = 512, chunk_overlap: int = 50) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len
    )
    chunks = []
    for doc in documents:
        page_num = doc.metadata["page"]
        doc_id = doc.metadata["document_id"]
        split_texts = splitter.split_text(doc.page_content)
        
        for idx, text in enumerate(split_texts):
            chunk_id = f"{doc_id}::page_{page_num}::chunk_{idx}"
            chunk_doc = Document(
                page_content=text,
                metadata={
                    "source": doc.metadata["source"],
                    "page": page_num,
                    "document_id": doc_id,
                    "chunk_id": chunk_id,
                    "chunk_index": idx,
                    "extraction_method": doc.metadata.get("extraction_method", "text"),
                    "content_type": doc.metadata.get("content_type", "text")
                }
            )
            chunks.append(chunk_doc)
    return chunks

def build_embeddings(model_id: str, device: str) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=model_id,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )

def build_faiss(chunks: List[Document], embeddings: HuggingFaceEmbeddings, index_path: str, force_rebuild: bool = False) -> FAISS:
    faiss_file = os.path.join(index_path, "index.faiss")
    pkl_file = os.path.join(index_path, "index.pkl")
    
    if not force_rebuild and os.path.exists(faiss_file) and os.path.exists(pkl_file):
        return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        
    os.makedirs(index_path, exist_ok=True)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(index_path)
    return vectorstore

def build_bm25(chunks: List[Document], persist_path: str, force_rebuild: bool = False) -> LocalBM25Retriever:
    if not force_rebuild and os.path.exists(persist_path):
        try:
            with open(persist_path, "rb") as f:
                data = pickle.load(f)
                if isinstance(data, LocalBM25Retriever):
                    return data
                elif isinstance(data, list):
                    return LocalBM25Retriever(data)
        except Exception:
            pass
            
    os.makedirs(os.path.dirname(os.path.abspath(persist_path)), exist_ok=True)
    retriever = LocalBM25Retriever(chunks)
    try:
        with open(persist_path, "wb") as f:
            pickle.dump(chunks, f)
    except Exception:
        pass
    return retriever
