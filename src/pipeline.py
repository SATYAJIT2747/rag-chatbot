import os
import time
from typing import Dict, Any, Optional
from langchain_community.vectorstores import FAISS
from .config import RAGConfig
from .models import ModelRegistry
from .indexer import LocalBM25Retriever
from .retrieval import dense_retrieve, bm25_retrieve, hybrid_retrieve, rewrite_query, rerank_documents
from .generation import build_context, generate_answer, format_cited_response
from .caching import RAGCacheManager, compute_doc_hash
from .metrics import MetricsTracker, calculate_llm_cost
from .guardrails import RAGGuardrailsManager

try:
    from langsmith import traceable
    from langsmith.run_helpers import get_current_run_tree
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def get_current_run_tree():
        return None

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
        cache_path = getattr(config, "cache_db_path", "artifacts/rag_cache.db")
        metrics_path = getattr(config, "metrics_db_path", "artifacts/rag_metrics.db")
        self.cache_manager = RAGCacheManager(db_path=cache_path)
        self.metrics_tracker = MetricsTracker(db_path=metrics_path)
        self.guardrails = RAGGuardrailsManager(config=config)

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

    @traceable(name="PDF RAG Execution", run_type="chain")
    def run(
        self,
        query: str,
        retrieval_mode: str = "hybrid",       # "hybrid", "dense", or "bm25"
        use_query_rewrite: bool = False,
        use_reranker: bool = True,
        return_parent_context: bool = True,
        retrieval_k: Optional[int] = None,
        rerank_k: Optional[int] = None,
        final_k: Optional[int] = None,
        user_id: str = "default_user"
    ) -> Dict[str, Any]:
        t_start = time.time()
        doc_hash = compute_doc_hash(self.config.pdf_path)
        document_id = os.path.splitext(os.path.basename(self.config.pdf_path))[0]
        model_name = self.config.llm_model_id
        prompt_version = getattr(self.config, "prompt_version", "v1.0")
        enable_cache = getattr(self.config, "enable_cache", True)
        cache_threshold = getattr(self.config, "semantic_cache_threshold", 0.90)
        pricing_map = getattr(self.config, "model_pricing", None)

        run_tree = get_current_run_tree()

        # ---------------------------------------------------------
        # 0. INPUT GUARDRAILS CHECK (Cheap Checks First)
        # ---------------------------------------------------------
        input_valid, reject_msg, status_code = self.guardrails.check_input(query)
        if not input_valid:
            latency_ms = round((time.time() - t_start) * 1000.0, 2)
            if run_tree:
                try:
                    tag_name = "prompt-injection" if status_code == "blocked_injection" else "length-exceeded"
                    run_tree.add_tags(["pdf-rag", "guardrail", "input-blocked", tag_name])
                    run_tree.add_metadata({
                        "document_id": document_id,
                        "model_name": model_name,
                        "guardrail_status": status_code,
                        "guardrail_blocked": True,
                        "prompt_version": prompt_version,
                        "user_id": user_id,
                        "llm_skipped": True
                    })
                except Exception:
                    pass

            # Log metrics for blocked request
            self.metrics_tracker.log_request(
                document_id=document_id,
                user_id=user_id,
                model_name=model_name,
                input_tokens=0,
                output_tokens=0,
                latency_ms=latency_ms,
                retrieval_latency_ms=0.0,
                llm_latency_ms=0.0,
                computed_cost_usd=0.0,
                cache_hit=False,
                cache_type="miss",
                request_category="pdf_qa",
                num_retrieved_chunks=0,
                token_count_type="actual",
                guardrail_status=status_code,
                grounding_status="skipped"
            )

            res = format_cited_response(
                query=query,
                rewritten_query=None,
                answer=reject_msg,
                retrieved_docs=[],
                latency_breakdown={"Total Latency (sec)": round(latency_ms / 1000.0, 3)},
                debug_payload={"guardrail_blocked": True, "status_code": status_code}
            )
            res["cache_type"] = "miss"
            res["cache_hit"] = False
            res["guardrail_blocked"] = True
            res["guardrail_status"] = status_code
            return res

        query_emb = None

        # ---------------------------------------------------------
        # 1. EXACT CACHE LOOKUP
        # ---------------------------------------------------------
        if enable_cache:
            exact_hit = self.cache_manager.get_exact(
                doc_hash=doc_hash,
                query=query,
                model_name=model_name,
                prompt_version=prompt_version
            )
            if exact_hit:
                latency_ms = round((time.time() - t_start) * 1000.0, 2)
                if run_tree:
                    try:
                        run_tree.add_tags(["pdf-rag", "cache-hit", "exact-cache"])
                        run_tree.add_metadata({
                            "document_id": document_id,
                            "model_name": model_name,
                            "cache_type": "exact",
                            "prompt_version": prompt_version,
                            "user_id": user_id,
                            "cache_hit": True,
                            "llm_skipped": True,
                            "request_category": "pdf_qa"
                        })
                    except Exception:
                        pass

                # Log metrics for exact cache hit
                self.metrics_tracker.log_request(
                    document_id=document_id,
                    user_id=user_id,
                    model_name=model_name,
                    input_tokens=0,
                    output_tokens=0,
                    latency_ms=latency_ms,
                    retrieval_latency_ms=0.0,
                    llm_latency_ms=0.0,
                    computed_cost_usd=0.0,
                    cache_hit=True,
                    cache_type="exact",
                    request_category="pdf_qa",
                    num_retrieved_chunks=0,
                    token_count_type="actual"
                )
                
                res = format_cited_response(
                    query=query,
                    rewritten_query=None,
                    answer=exact_hit["answer"],
                    retrieved_docs=[],
                    latency_breakdown={"Total Latency (sec)": round(latency_ms / 1000.0, 3)},
                    debug_payload={"cache_hit": True, "cache_type": "exact", "llm_skipped": True}
                )
                res["cache_type"] = "exact"
                res["cache_hit"] = True
                res["sources"] = exact_hit.get("sources", [])
                res["citations"] = exact_hit.get("citations", [])
                return res

            # ---------------------------------------------------------
            # 2. SEMANTIC CACHE LOOKUP
            # ---------------------------------------------------------
            try:
                embedder = self.registry.load_embedding_model()
                query_emb = embedder.encode(query).tolist()
                
                semantic_hit = self.cache_manager.get_semantic(
                    doc_hash=doc_hash,
                    query=query,
                    query_embedding=query_emb,
                    model_name=model_name,
                    prompt_version=prompt_version,
                    threshold=cache_threshold
                )
                if semantic_hit:
                    latency_ms = round((time.time() - t_start) * 1000.0, 2)
                    if run_tree:
                        try:
                            run_tree.add_tags(["pdf-rag", "cache-hit", "semantic-cache"])
                            run_tree.add_metadata({
                                "document_id": document_id,
                                "model_name": model_name,
                                "cache_type": "semantic",
                                "prompt_version": prompt_version,
                                "user_id": user_id,
                                "cache_hit": True,
                                "llm_skipped": True,
                                "similarity": semantic_hit.get("similarity"),
                                "request_category": "pdf_qa"
                            })
                        except Exception:
                            pass
                    self.metrics_tracker.log_request(
                        document_id=document_id,
                        user_id=user_id,
                        model_name=model_name,
                        input_tokens=0,
                        output_tokens=0,
                        latency_ms=latency_ms,
                        retrieval_latency_ms=0.0,
                        llm_latency_ms=0.0,
                        computed_cost_usd=0.0,
                        cache_hit=True,
                        cache_type="semantic",
                        request_category="pdf_qa",
                        num_retrieved_chunks=0,
                        token_count_type="actual"
                    )
                    
                    res = format_cited_response(
                        query=query,
                        rewritten_query=None,
                        answer=semantic_hit["answer"],
                        retrieved_docs=[],
                        latency_breakdown={"Total Latency (sec)": round(latency_ms / 1000.0, 3)},
                        debug_payload={"cache_hit": True, "cache_type": "semantic", "similarity": semantic_hit.get("similarity"), "llm_skipped": True}
                    )
                    res["cache_type"] = "semantic"
                    res["cache_hit"] = True
                    res["sources"] = semantic_hit.get("sources", [])
                    res["citations"] = semantic_hit.get("citations", [])
                    return res
            except Exception as e:
                print(f"[RAGPipeline] Warning: Embedding generation for semantic cache error: {e}")

        # ---------------------------------------------------------
        # 3. LIVE RAG PIPELINE EXECUTION (CACHE MISS)
        # ---------------------------------------------------------
        if run_tree:
            try:
                run_tree.add_tags(["pdf-rag", "cache-miss"])
                run_tree.add_metadata({
                    "document_id": document_id,
                    "model_name": model_name,
                    "cache_type": "miss",
                    "prompt_version": prompt_version,
                    "user_id": user_id,
                    "cache_hit": False,
                    "llm_skipped": False,
                    "request_category": "pdf_qa"
                })
            except Exception:
                pass
        k_ret = retrieval_k or self.config.retrieval_k
        k_rr = rerank_k or self.config.rerank_k
        k_fin = final_k or self.config.final_k
        
        # 3a. Query Rewrite
        t_qr_0 = time.time()
        rewritten_q = None
        search_query = query
        if use_query_rewrite:
            rewritten_q = rewrite_query(query, self.registry)
            search_query = rewritten_q
        qr_ms = round((time.time() - t_qr_0) * 1000.0, 2)
        
        # 3b. Candidate Retrieval
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
        
        # 3c. Cross-Encoder Reranking
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
            
        retrieval_total_ms = round(ret_ms + rerank_ms + qr_ms, 2)

        # ---------------------------------------------------------
        # 3d. PDF SCOPE GUARDRAIL CHECK (Reuses Retrieved Results)
        # ---------------------------------------------------------
        in_scope, scope_msg = self.guardrails.check_scope(final_docs, debug_info)
        if not in_scope:
            total_sec = round(time.time() - t_start, 3)
            total_ms = round(total_sec * 1000.0, 2)
            if run_tree:
                try:
                    run_tree.add_tags(["pdf-rag", "guardrail", "out-of-scope"])
                    run_tree.add_metadata({
                        "document_id": document_id,
                        "model_name": model_name,
                        "guardrail_status": "blocked_scope",
                        "guardrail_blocked": True,
                        "prompt_version": prompt_version,
                        "user_id": user_id,
                        "llm_skipped": True
                    })
                except Exception:
                    pass

            self.metrics_tracker.log_request(
                document_id=document_id,
                user_id=user_id,
                model_name=model_name,
                input_tokens=0,
                output_tokens=0,
                latency_ms=total_ms,
                retrieval_latency_ms=retrieval_total_ms,
                llm_latency_ms=0.0,
                computed_cost_usd=0.0,
                cache_hit=False,
                cache_type="miss",
                request_category="pdf_qa",
                num_retrieved_chunks=len(final_docs),
                token_count_type="actual",
                guardrail_status="blocked_scope",
                grounding_status="skipped"
            )

            res = format_cited_response(
                query=query,
                rewritten_query=rewritten_q,
                answer=scope_msg,
                retrieved_docs=final_docs,
                latency_breakdown={"Total Latency (sec)": total_sec},
                debug_payload={"guardrail_blocked": True, "status_code": "blocked_scope"}
            )
            res["cache_type"] = "miss"
            res["cache_hit"] = False
            res["guardrail_blocked"] = True
            res["guardrail_status"] = "blocked_scope"
            return res

        # 3e. Context Construction
        context = build_context(
            retrieved_chunks=final_docs,
            page_map=self.page_map,
            return_parent_context=return_parent_context,
            max_context_chars=self.config.max_parent_context_chars
        )
        debug_info["context_used"] = context
        
        # 3f. Generation
        gen_res = generate_answer(
            query=query,
            context=context,
            registry=self.registry,
            max_new_tokens=self.config.llm_max_new_tokens,
            temperature=self.config.llm_temperature
        )
        gen_sec = gen_res["latency_sec"]
        llm_ms = round(gen_sec * 1000.0, 2)
        raw_answer = gen_res["answer"]

        # ---------------------------------------------------------
        # 3g. OUTPUT GROUNDING VALIDATION
        # ---------------------------------------------------------
        grounding_res = self.guardrails.check_output_grounding(query, context, raw_answer)
        final_answer = raw_answer
        guardrail_status = "allowed"
        grounding_status = "grounded"

        if not grounding_res["grounded"]:
            final_answer = "The provided document does not contain enough information to answer this."
            guardrail_status = "blocked_ungrounded"
            grounding_status = "ungrounded"
            if run_tree:
                try:
                    run_tree.add_tags(["pdf-rag", "guardrail", "grounding-fail"])
                except Exception:
                    pass

        total_sec = round(time.time() - t_start, 3)
        total_ms = round(total_sec * 1000.0, 2)
        
        input_tokens = gen_res.get("input_tokens", 0)
        output_tokens = gen_res.get("output_tokens", 0)
        token_count_type = gen_res.get("token_count_type", "estimated")
        
        cost_usd = calculate_llm_cost(
            model_name=model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            pricing_map=pricing_map
        )
        
        latency_breakdown = {
            "Query Rewrite (ms)": qr_ms if use_query_rewrite else 0.0,
            f"Retrieval [{retrieval_mode.upper()}] (ms)": ret_ms,
            "Cross-Encoder Rerank (ms)": rerank_ms if use_reranker else 0.0,
            "LLM Generation (sec)": gen_sec,
            "Total Latency (sec)": total_sec
        }
        
        formatted_res = format_cited_response(
            query=query,
            rewritten_query=rewritten_q,
            answer=final_answer,
            retrieved_docs=final_docs,
            latency_breakdown=latency_breakdown,
            debug_payload=debug_info
        )
        
        # Log request to persistent metrics DB
        self.metrics_tracker.log_request(
            document_id=document_id,
            user_id=user_id,
            model_name=model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=total_ms,
            retrieval_latency_ms=retrieval_total_ms,
            llm_latency_ms=llm_ms,
            computed_cost_usd=cost_usd,
            cache_hit=False,
            cache_type="miss",
            request_category="pdf_qa",
            num_retrieved_chunks=len(final_docs),
            token_count_type=token_count_type,
            guardrail_status=guardrail_status,
            grounding_status=grounding_status
        )
        
        # Write to Exact & Semantic Cache ONLY if request was grounded and in scope
        if enable_cache and guardrail_status == "allowed":
            if query_emb is None:
                try:
                    embedder = self.registry.load_embedding_model()
                    query_emb = embedder.encode(query).tolist()
                except Exception:
                    query_emb = None

            self.cache_manager.put(
                doc_hash=doc_hash,
                document_id=document_id,
                query=query,
                query_embedding=query_emb,
                model_name=model_name,
                prompt_version=prompt_version,
                answer=formatted_res["answer"],
                citations=formatted_res.get("citations", []),
                sources=formatted_res.get("sources", [])
            )
            
        formatted_res["cache_type"] = "miss"
        formatted_res["cache_hit"] = False
        formatted_res["input_tokens"] = input_tokens
        formatted_res["output_tokens"] = output_tokens
        formatted_res["cost_usd"] = cost_usd
        formatted_res["guardrail_status"] = guardrail_status
        formatted_res["grounding_status"] = grounding_status
        return formatted_res

