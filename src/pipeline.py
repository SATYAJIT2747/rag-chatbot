import time
from typing import Dict, Any, Optional
from langchain_community.vectorstores import FAISS
from .config import RAGConfig
from .models import ModelRegistry
from .indexer import LocalBM25Retriever
from .retrieval import dense_retrieve, bm25_retrieve, hybrid_retrieve, rewrite_query, rerank_documents
from .generation import build_context, generate_answer, format_cited_response

class RAGPipeline:
    def __init__(
        self,
        config: RAGConfig,
        registry: ModelRegistry,
        vectorstore: FAISS,
        bm25_retriever: LocalBM25Retriever,
        page_map: Dict[int, str]
    ):
        self.config = config
        self.registry = registry
        self.vectorstore = vectorstore
        self.bm25_retriever = bm25_retriever
        self.page_map = page_map

    @classmethod
    def from_defaults(cls, config: Optional[RAGConfig] = None, force_rebuild: bool = False) -> "RAGPipeline":
        from .config import get_system_device_info
        from .document_loader import load_and_preprocess_pdf
        from .indexer import chunk_documents, build_embeddings, build_faiss, build_bm25
        
        cfg = config or RAGConfig()
        dev_info = get_system_device_info(cfg)
        device = dev_info["device"]
        
        registry = ModelRegistry(cfg, device=device)
        docs, page_map = load_and_preprocess_pdf(cfg.pdf_path)
        chunks = chunk_documents(docs, chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap)
        
        embeddings = build_embeddings(cfg.embedding_model_id, device=device)
        vectorstore = build_faiss(chunks, embeddings, cfg.faiss_dir, force_rebuild=force_rebuild)
        bm25_retriever = build_bm25(chunks, cfg.bm25_path, force_rebuild=force_rebuild)
        
        return cls(cfg, registry, vectorstore, bm25_retriever, page_map)

    def query(self, query: str, **kwargs) -> Dict[str, Any]:
        return self.run(query, **kwargs)

    def run(
        self,
        query: str,
        retrieval_mode: str = "hybrid",       # "hybrid", "dense", or "bm25"
        use_query_rewrite: bool = False,
        use_reranker: bool = True,
        return_parent_context: bool = True,
        retrieval_k: Optional[int] = None,
        rerank_k: Optional[int] = None,
        final_k: Optional[int] = None
    ) -> Dict[str, Any]:
        t_start = time.time()
        k_ret = retrieval_k or self.config.retrieval_k
        k_rr = rerank_k or self.config.rerank_k
        k_fin = final_k or self.config.final_k
        
        # 1. Query Rewrite
        t_qr_0 = time.time()
        rewritten_q = None
        search_query = query
        if use_query_rewrite:
            rewritten_q = rewrite_query(query, self.registry)
            search_query = rewritten_q
        qr_ms = round((time.time() - t_qr_0) * 1000.0, 2)
        
        # 2. Candidate Retrieval
        t_ret_0 = time.time()
        debug_info = {}
        
        if retrieval_mode == "dense":
            dense_res = dense_retrieve(search_query, self.vectorstore, top_k=k_ret)
            candidate_docs = [doc for doc, score, rank in dense_res]
            debug_info["mode"] = "dense"
            debug_info["raw_results"] = dense_res
        elif retrieval_mode == "bm25":
            bm25_res = bm25_retrieve(search_query, self.bm25_retriever, top_k=k_ret)
            candidate_docs = [doc for doc, score, rank in bm25_res]
            debug_info["mode"] = "bm25"
            debug_info["raw_results"] = bm25_res
        else:
            hybrid_res = hybrid_retrieve(search_query, self.vectorstore, self.bm25_retriever, top_k=k_ret, rrf_k=self.config.rrf_k)
            candidate_docs = hybrid_res["candidates"]
            debug_info["mode"] = "hybrid"
            debug_info["rrf_debug"] = hybrid_res["debug_records"]
            debug_info["dense_raw"] = hybrid_res["dense_raw"]
            debug_info["bm25_raw"] = hybrid_res["bm25_raw"]
            
        ret_ms = round((time.time() - t_ret_0) * 1000.0, 2)
        
        # 3. Cross-Encoder Reranking
        t_rr_0 = time.time()
        rerank_ms = 0.0
        if use_reranker and candidate_docs:
            rerank_res = rerank_documents(
                query=search_query,
                candidate_docs=candidate_docs[:k_rr],
                registry=self.registry,
                top_k=k_fin
            )
            final_docs = rerank_res["reranked_docs"]
            rerank_ms = rerank_res["latency_ms"]
            debug_info["rerank_debug"] = rerank_res["rerank_debug"]
        else:
            final_docs = candidate_docs[:k_fin]
            
        # 4. Context Construction
        context = build_context(
            retrieved_chunks=final_docs,
            page_map=self.page_map,
            return_parent_context=return_parent_context,
            max_context_chars=self.config.max_parent_context_chars
        )
        debug_info["context_used"] = context
        
        # 5. Generation
        gen_res = generate_answer(
            query=query,
            context=context,
            registry=self.registry,
            max_new_tokens=self.config.llm_max_new_tokens,
            temperature=self.config.llm_temperature
        )
        gen_sec = gen_res["latency_sec"]
        
        total_sec = round(time.time() - t_start, 3)
        
        latency_breakdown = {
            "Query Rewrite (ms)": qr_ms if use_query_rewrite else 0.0,
            f"Retrieval [{retrieval_mode.upper()}] (ms)": ret_ms,
            "Cross-Encoder Rerank (ms)": rerank_ms if use_reranker else 0.0,
            "LLM Generation (sec)": gen_sec,
            "Total Latency (sec)": total_sec
        }
        
        return format_cited_response(
            query=query,
            rewritten_query=rewritten_q,
            answer=gen_res["answer"],
            retrieved_docs=final_docs,
            latency_breakdown=latency_breakdown,
            debug_payload=debug_info
        )
