import os
import sys
import json
import argparse
from tabulate import tabulate

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import RAGConfig, get_system_device_info
from src.pipeline import RAGPipeline
from src.evaluator import DEFAULT_EVAL_SUITE, evaluate_retrieval_suite

def print_header(title: str):
    print("\n" + "=" * 65)
    print(f"  {title.upper()}")
    print("=" * 65)

def run_cli():
    parser = argparse.ArgumentParser(description="Production-Grade Advanced RAG PDF Bot")
    parser.add_argument("--pdf", type=str, default=None, help="Path to PDF document (default: sample_document.pdf)")
    parser.add_argument("--query", type=str, default=None, help="User question to answer")
    parser.add_argument("--mode", type=str, choices=["hybrid", "dense", "bm25"], default="hybrid", help="Retrieval mode")
    parser.add_argument("--no-rerank", action="store_true", help="Disable cross-encoder reranker")
    parser.add_argument("--rewrite", action="store_true", help="Enable LLM query rewriting")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild vector and BM25 index")
    parser.add_argument("--eval", action="store_true", help="Run benchmark evaluation suite")
    
    args = parser.parse_args()
    
    config = RAGConfig()
    if args.pdf:
        config.pdf_path = args.pdf
        
    dev_info = get_system_device_info(config)
    
    print_header("1. System Diagnostics & Configuration")
    print(f"Execution Device : {dev_info['device'].upper()} ({dev_info['gpu_name']})")
    print(f"CUDA Available   : {dev_info['cuda_available']}")
    print(f"System RAM       : {dev_info['ram_gb']} GB")
    print(f"Target Document  : {os.path.abspath(config.pdf_path)}")
    print(f"Dense Embeddings : {config.embedding_model_id}")
    print(f"Reranker Model   : {config.reranker_model_id}")
    print(f"LLM Model        : {config.llm_model_id}")
    
    print_header("2. Initializing Advanced RAG Pipeline")
    print("Loading document, applying Vision OCR fallback if needed, & building hybrid indexes...")
    pipeline = RAGPipeline.from_defaults(config=config, force_rebuild=args.rebuild)
    print("[SUCCESS] Pipeline initialized successfully.")
    
    # Demonstration / User Queries
    if args.query:
        queries = [args.query]
    else:
        queries = [
            "What is the primary limitation of basic vector similarity search in RAG?",
            "How does Hybrid Search combine BM25 and dense vector search using RRF?",
            "What is the company policy on employee vacation in Miami?"
        ]
        
    print_header("3. Query Execution & Grounded Generation")
    demo_results = []
    for q in queries:
        print(f"\n[QUERY]: '{q}'")
        res = pipeline.run(
            query=q,
            retrieval_mode=args.mode,
            use_query_rewrite=args.rewrite,
            use_reranker=not args.no_rerank,
            return_parent_context=config.return_parent_context
        )
        demo_results.append(res)
        
        print(f"ANSWER    : {res['answer']}")
        print(f"CITATIONS : {res['citations']}")
        
        table_latency = [[k, f"{v} {'sec' if 'sec' in k.lower() else 'ms'}"] for k, v in res["latency_breakdown"].items()]
        print("\nLatency Profile:")
        print(tabulate(table_latency, headers=["Pipeline Stage", "Execution Time"], tablefmt="grid"))
        print("-" * 65)
        
    # Run evaluation suite if requested or in default demo mode
    if args.eval or not args.query:
        print_header("4. Retrieval Benchmark Evaluation")
        eval_res = evaluate_retrieval_suite(
            eval_suite=DEFAULT_EVAL_SUITE,
            vectorstore=pipeline.vectorstore,
            bm25_ret=pipeline.bm25_retriever,
            registry=pipeline.registry
        )
        
        eval_rows = []
        for method, metrics in eval_res["methods"].items():
            eval_rows.append([
                method,
                metrics.get("Recall@5", 0.0),
                metrics.get("Recall@10", 0.0),
                metrics.get("MRR@10", 0.0),
                metrics.get("nDCG@10", 0.0)
            ])
            
        print(tabulate(eval_rows, headers=["Method", "Recall@5", "Recall@10", "MRR@10", "nDCG@10"], tablefmt="grid"))
        
        # Save persistent JSON report
        os.makedirs(config.report_dir, exist_ok=True)
        report_file = os.path.join(config.report_dir, "demo_report.json")
        report_data = {
            "configuration": {
                "pdf_path": config.pdf_path,
                "device": dev_info["device"],
                "embedding_model": config.embedding_model_id,
                "reranker_model": config.reranker_model_id,
                "llm_model": config.llm_model_id
            },
            "eval_results": eval_res,
            "demo_runs": demo_results
        }
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)
        print(f"\n[REPORT] Saved benchmark evaluation report to: {report_file}")

    print_header("Execution Complete")
    print("To launch the interactive Web UI, run:")
    print("  streamlit run app.py")

if __name__ == "__main__":
    run_cli()
