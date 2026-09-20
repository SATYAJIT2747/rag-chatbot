import time
import torch
from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from .indexer import LocalBM25Retriever
from .models import ModelRegistry

def dense_retrieve(query: str, vectorstore: FAISS, top_k: int = 20) -> List[Tuple[Document, float, int]]:
    docs_and_scores = vectorstore.similarity_search_with_score(query, k=top_k)
    ranked = []
    for rank, (doc, score) in enumerate(docs_and_scores, 1):
        ranked.append((doc, float(score), rank))
    return ranked

def bm25_retrieve(query: str, retriever: LocalBM25Retriever, top_k: int = 20) -> List[Tuple[Document, float, int]]:
    return retriever.retrieve(query, top_k=top_k)

def hybrid_retrieve(
    query: str,
    vectorstore: FAISS,
    bm25_ret: LocalBM25Retriever,
    top_k: int = 20,
    rrf_k: int = 60
) -> Dict[str, Any]:
    dense_results = dense_retrieve(query, vectorstore, top_k=top_k)
    bm25_results = bm25_retrieve(query, bm25_ret, top_k=top_k)
    
    rrf_scores = {}
    doc_map = {}
    dense_rank_map = {}
    bm25_rank_map = {}
    
    for doc, _, rank in dense_results:
        cid = doc.metadata["chunk_id"]
        doc_map[cid] = doc
        dense_rank_map[cid] = rank
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (rrf_k + rank))
        
    for doc, _, rank in bm25_results:
        cid = doc.metadata["chunk_id"]
        doc_map[cid] = doc
        bm25_rank_map[cid] = rank
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (rrf_k + rank))
        
    sorted_chunk_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)[:top_k]
    
    fused_candidates = []
    debug_records = []
    
    for final_rank, cid in enumerate(sorted_chunk_ids, 1):
        doc = doc_map[cid]
        fused_candidates.append(doc)
        debug_records.append({
            "final_rank": final_rank,
            "chunk_id": cid,
            "page": doc.metadata["page"],
            "rrf_score": round(rrf_scores[cid], 5),
            "dense_rank": dense_rank_map.get(cid, "-"),
            "bm25_rank": bm25_rank_map.get(cid, "-"),
            "preview": doc.page_content[:90].replace('\n', ' ') + "..."
        })
        
    return {
        "candidates": fused_candidates,
        "debug_records": debug_records,
        "dense_raw": dense_results,
        "bm25_raw": bm25_results
    }

def rewrite_query(query: str, registry: ModelRegistry) -> str:
    llm, tokenizer = registry.load_llm()
    
    if hasattr(llm, "invoke") and tokenizer is None:
        sys_msg = "You are an expert query reformulation assistant for information retrieval. Given a user question, rewrite it into ONE concise, search-optimized query. Preserve original intent. Do NOT answer the question. Output ONLY the rewritten search query and nothing else."
        messages = [
            ("system", sys_msg),
            ("user", f"Original Query: {query}\nRewritten search query:")
        ]
        response = llm.invoke(messages)
        content = response.content if hasattr(response, "content") else str(response)
        if isinstance(content, list):
            rewritten = "".join([part.get("text", "") if isinstance(part, dict) else str(part) for part in content]).strip()
        else:
            rewritten = str(content).strip()
        return rewritten.split('\n')[0].strip('"')
        
    prompt = f"""<|im_start|>system
You are an expert query reformulation assistant for information retrieval.
Given a user question, rewrite it into ONE concise, search-optimized query.
Rules:
- Preserve the original intent.
- Do NOT answer the question.
- Do NOT hallucinate facts.
- Output ONLY the rewritten search query and nothing else.<|im_end|>
<|im_start|>user
Original Query: {query}<|im_end|>
<|im_start|>assistant
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(registry.device)
    with torch.no_grad():
        output_ids = llm.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    rewritten = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return rewritten.split('\n')[0].strip('"')

def rerank_documents(
    query: str,
    candidate_docs: List[Document],
    registry: ModelRegistry,
    top_k: int = 5,
    batch_size: int = 8
) -> Dict[str, Any]:
    if not candidate_docs:
        return {"reranked_docs": [], "scores": [], "latency_ms": 0.0, "rerank_debug": []}
        
    t0 = time.time()
    reranker_model, reranker_tokenizer = registry.load_reranker_model()
    
    pairs = [[query, doc.page_content] for doc in candidate_docs]
    all_scores = []
    
    for i in range(0, len(pairs), batch_size):
        batch_pairs = pairs[i:i + batch_size]
        inputs = reranker_tokenizer(
            batch_pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(registry.device)
        
        with torch.no_grad():
            outputs = reranker_model(**inputs)
            scores = outputs.logits.squeeze(-1).cpu().tolist()
            if isinstance(scores, float):
                scores = [scores]
            all_scores.extend(scores)
            
    doc_score_pairs = list(zip(candidate_docs, all_scores))
    doc_score_pairs_ranked = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
    
    rerank_debug = []
    for new_rank, (doc, score) in enumerate(doc_score_pairs_ranked, 1):
        old_rank = [d for d, s in doc_score_pairs].index(doc) + 1
        rerank_debug.append({
            "chunk_id": doc.metadata.get("chunk_id", "unknown"),
            "page": doc.metadata.get("page", 0),
            "before_rank": old_rank,
            "after_rank": new_rank,
            "reranker_score": round(float(score), 4)
        })
        
    top_results = doc_score_pairs_ranked[:top_k]
    latency_ms = (time.time() - t0) * 1000.0
    
    return {
        "reranked_docs": [doc for doc, score in top_results],
        "scores": [score for doc, score in top_results],
        "latency_ms": round(latency_ms, 2),
        "rerank_debug": rerank_debug
    }
