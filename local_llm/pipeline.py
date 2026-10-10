import os
import copy
from typing import Optional
from src.config import RAGConfig
from src.pipeline import RAGPipeline
from .llm_client import client_from_env


def build_local_config(base: Optional[RAGConfig] = None) -> RAGConfig:
    """Same settings as the Gemini pipeline; only the LLM and its cache/metrics DBs differ."""
    cfg = copy.deepcopy(base) if base else RAGConfig()
    cfg.llm_model_id = os.getenv("LOCAL_LLM_MODEL", "qwen3.5-4b-q8")
    cfg.cache_db_path = "artifacts/local_llm/rag_cache.db"
    cfg.metrics_db_path = "artifacts/local_llm/rag_metrics.db"
    cfg.report_dir = "artifacts/local_llm/report"
    cfg.model_pricing[cfg.llm_model_id] = {"input_per_1m": 0.00, "output_per_1m": 0.00}
    return cfg


def attach_local_llm(pipeline: RAGPipeline) -> RAGPipeline:
    """Pre-loads the llama-server client so ModelRegistry.load_llm() never touches Gemini/HF."""
    pipeline.registry.llm_model = client_from_env(
        temperature=pipeline.config.llm_temperature,
        max_tokens=pipeline.config.llm_max_new_tokens
    )
    pipeline.registry.llm_tokenizer = None
    return pipeline


def build_local_pipeline(config: Optional[RAGConfig] = None, force_rebuild: bool = False) -> RAGPipeline:
    cfg = build_local_config(config)
    return attach_local_llm(RAGPipeline.from_defaults(config=cfg, force_rebuild=force_rebuild))


def local_pipeline_from(gemini_pipeline: RAGPipeline) -> RAGPipeline:
    """
    Builds the local pipeline on top of an existing Gemini pipeline's indexes and
    embedding/reranker models, so both answer from the exact same retrieved context.
    """
    gemini_pipeline.registry.load_embedding_model()
    gemini_pipeline.registry.load_reranker_model()

    cfg = build_local_config(gemini_pipeline.config)
    registry = copy.copy(gemini_pipeline.registry)
    registry.config = cfg
    local = RAGPipeline(cfg, registry, gemini_pipeline.vectorstore, gemini_pipeline.bm25_retriever, gemini_pipeline.page_map)
    return attach_local_llm(local)
