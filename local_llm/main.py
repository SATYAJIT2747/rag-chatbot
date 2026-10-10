import os
import sys
import argparse
from tabulate import tabulate

# Run from pdf_rag_bot/ so src/ imports and relative artifact paths resolve
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from src.config import RAGConfig
from local_llm.llm_client import LlamaServerChat
from local_llm.pipeline import build_local_pipeline


def run_cli():
    parser = argparse.ArgumentParser(description="PDF RAG Bot with a local llama-server LLM (Qwen3.5)")
    parser.add_argument("--pdf", type=str, default=None, help="Path to PDF document")
    parser.add_argument("--query", type=str, required=True, help="User question to answer")
    parser.add_argument("--mode", type=str, choices=["hybrid", "dense", "bm25"], default="hybrid", help="Retrieval mode")
    parser.add_argument("--no-rerank", action="store_true", help="Disable cross-encoder reranker")
    parser.add_argument("--rewrite", action="store_true", help="Enable LLM query rewriting")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild vector and BM25 index")
    args = parser.parse_args()

    config = RAGConfig()
    if args.pdf:
        config.pdf_path = args.pdf

    pipeline = build_local_pipeline(config=config, force_rebuild=args.rebuild)
    print(f"LLM Model        : {pipeline.config.llm_model_id}")
    llm: LlamaServerChat = pipeline.registry.llm_model  # type: ignore[assignment]
    print(f"llama-server GGUF: {llm.served_model()}")

    res = pipeline.run(
        query=args.query,
        retrieval_mode=args.mode,
        use_query_rewrite=args.rewrite,
        use_reranker=not args.no_rerank,
        return_parent_context=config.return_parent_context
    )
    print(f"\nANSWER    : {res['answer']}")
    print(f"CITATIONS : {res['citations']}")
    table_latency = [[k, f"{v} {'sec' if 'sec' in k.lower() else 'ms'}"] for k, v in res["latency_breakdown"].items()]
    print(tabulate(table_latency, headers=["Pipeline Stage", "Execution Time"], tablefmt="grid"))


if __name__ == "__main__":
    run_cli()
