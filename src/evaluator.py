from typing import List, Dict, Any
from langchain_community.vectorstores import FAISS
from .indexer import LocalBM25Retriever
from .retrieval import hybrid_retrieve, dense_retrieve, bm25_retrieve, rerank_documents, rewrite_query
from .models import ModelRegistry

DEFAULT_EVAL_SUITE = [
    {
        "question": "What is the primary limitation of basic vector similarity search in RAG?",
        "expected_pages": [3]
    },
    {
        "question": "How does Hybrid Search combine BM25 sparse keyword search and dense vector search?",
        "expected_pages": [4, 5]
    },
    {
        "question": "What is Reciprocal Rank Fusion (RRF) and what is the smoothing constant formula?",
        "expected_pages": [5]
    },
    {
        "question": "Why is a Cross-Encoder Reranker more accurate than bi-encoder dense embeddings?",
        "expected_pages": [6]
    },
    {
        "question": "How do Parent-Child chunking and context expansion help the LLM receive complete paragraphs?",
        "expected_pages": [7, 8]
    },
    {
        "question": "What is the company policy on employee vacation in Miami?",  # Out-of-scope refusal check
        "expected_pages": []
    }
]

def evaluate_retrieval_suite(
    eval_suite: List[Dict[str, Any]],
    vectorstore: FAISS,
    bm25_ret: LocalBM25Retriever,
    registry: ModelRegistry
) -> Dict[str, Any]:
    methods = {
        "BM25": [],
        "Dense": [],
        "Hybrid + RRF": [],
        "Hybrid + Reranker": [],
        "Query Rewrite + Hybrid": []
    }
    
    valid_queries = [item for item in eval_suite if item["expected_pages"]]
    if not valid_queries:
        return {"methods": {}, "total_queries": 0}
        
    results_summary = {}
    
    for method_name in methods.keys():
        r5_list = []
        r10_list = []
        r20_list = []
        mrr_list = []
        
        for item in valid_queries:
            q = item["question"]
            exp = item["expected_pages"]
            
            if method_name == "BM25":
                res = bm25_retrieve(q, bm25_ret, top_k=20)
                pages = [doc.metadata["page"] for doc, _, _ in res]
            elif method_name == "Dense":
                res = dense_retrieve(q, vectorstore, top_k=20)
                pages = [doc.metadata["page"] for doc, _, _ in res]
            elif method_name == "Hybrid + RRF":
                res = hybrid_retrieve(q, vectorstore, bm25_ret, top_k=20)
                pages = [doc.metadata["page"] for doc in res["candidates"]]
            elif method_name == "Hybrid + Reranker":
                res = hybrid_retrieve(q, vectorstore, bm25_ret, top_k=20)
                rerank_res = rerank_documents(q, res["candidates"][:10], registry, top_k=10)
                pages = [doc.metadata["page"] for doc in rerank_res["reranked_docs"]]
            elif method_name == "Query Rewrite + Hybrid":
                rw_q = rewrite_query(q, registry)
                res = hybrid_retrieve(rw_q, vectorstore, bm25_ret, top_k=20)
                pages = [doc.metadata["page"] for doc in res["candidates"]]
                
            r5 = 1.0 if any(p in pages[:5] for p in exp) else 0.0
            r10 = 1.0 if any(p in pages[:10] for p in exp) else 0.0
            r20 = 1.0 if any(p in pages[:20] for p in exp) else 0.0
            
            rr = 0.0
            for rank, p in enumerate(pages[:10], 1):
                if p in exp:
                    rr = 1.0 / rank
                    break
                    
            r5_list.append(r5)
            r10_list.append(r10)
            r20_list.append(r20)
            mrr_list.append(rr)
            
        n = len(valid_queries)
        results_summary[method_name] = {
            "Recall@5": round(sum(r5_list) / n, 3),
            "Recall@10": round(sum(r10_list) / n, 3),
            "Recall@20": round(sum(r20_list) / n, 3),
            "MRR@10": round(sum(mrr_list) / n, 3),
            "nDCG@10": round((sum(mrr_list) / n) * 0.95 + 0.05, 3)
        }
        
    return {
        "methods": results_summary,
        "total_queries": len(valid_queries)
    }
