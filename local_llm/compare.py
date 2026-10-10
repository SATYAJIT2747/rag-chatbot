"""
Runs the same questions through the Gemini pipeline and the local llama-server
pipeline (identical retrieval, reranking, context and guardrails) and writes a
side-by-side report to artifacts/local_llm/report/.

Usage (from pdf_rag_bot/):
    ..\\.venv\\Scripts\\python.exe -m local_llm.compare
"""
import os
import sys
import json
import time
import argparse
from tabulate import tabulate

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from src.config import RAGConfig
from src.pipeline import RAGPipeline
from src.evaluator import gold_suite_for
from local_llm.llm_client import LlamaServerChat
from local_llm.pipeline import local_pipeline_from


def run_one(pipeline: RAGPipeline, question: str) -> dict:
    t0 = time.time()
    try:
        res = pipeline.run(query=question, retrieval_mode="hybrid", use_reranker=True)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "wall_sec": round(time.time() - t0, 3)}
    llm_sec = res["latency_breakdown"].get("LLM Generation (sec)", 0.0)
    out_tok = res.get("output_tokens", 0)
    return {
        "answer": res["answer"],
        "citations": res["citations"],
        "guardrail_status": res.get("guardrail_status"),
        "grounding_status": res.get("grounding_status", "skipped"),
        "llm_sec": llm_sec,
        "total_sec": res["latency_breakdown"].get("Total Latency (sec)", 0.0),
        "input_tokens": res.get("input_tokens", 0),
        "output_tokens": out_tok,
        "tokens_per_sec": round(out_tok / llm_sec, 1) if llm_sec else 0.0,
        "cost_usd": res.get("cost_usd", 0.0)
    }


def citation_hit(expected: list, cited: list) -> str:
    if not expected:
        return "n/a"
    return "yes" if set(expected) & set(cited) else "no"


def main():
    parser = argparse.ArgumentParser(description="Compare Gemini vs local llama-server generation")
    parser.add_argument("--pdf", type=str, default=None, help="Path to PDF document")
    parser.add_argument("--questions", type=str, default=None,
                        help="JSON list of {question, expected_pages} (default: gold dataset questions for this PDF)")
    args = parser.parse_args()

    config = RAGConfig()
    if args.pdf:
        config.pdf_path = args.pdf

    eval_suite = gold_suite_for(config.pdf_path)
    if args.questions:
        with open(args.questions, encoding="utf-8") as f:
            eval_suite = json.load(f)
    # Caching would short-circuit generation and hide real latency
    config.enable_cache = False
    # src/indexer reuses whatever index is on disk regardless of PDF, so keep one per document
    index_dir = os.path.join("artifacts", "local_llm", "index", os.path.splitext(os.path.basename(config.pdf_path))[0])
    config.faiss_dir = os.path.join(index_dir, "faiss")
    config.bm25_path = os.path.join(index_dir, "bm25.pkl")

    gemini = RAGPipeline.from_defaults(config=config)
    local = local_pipeline_from(gemini)
    llm: LlamaServerChat = local.registry.llm_model  # type: ignore[assignment]
    served = llm.served_model()
    print(f"Gemini model : {gemini.config.llm_model_id}")
    print(f"Local model  : {local.config.llm_model_id} ({served})")

    # Warm up the local server so model load / first-prompt overhead isn't billed to question 1
    llm.invoke([("user", "Say OK.")])

    rows = []
    for item in eval_suite:
        q = item["question"]
        print(f"\n[QUERY] {q}")
        g = run_one(gemini, q)
        l = run_one(local, q)
        rows.append({"question": q, "expected_pages": item["expected_pages"], "gemini": g, "local": l})
        for name, r in (("gemini", g), ("local", l)):
            print(f"  {name:6}: {r.get('error') or r['answer'][:160]!r}")

    summary_rows = []
    for name in ("gemini", "local"):
        ok = [r[name] for r in rows if "error" not in r[name]]
        gen = [r for r in ok if r["guardrail_status"] != "blocked_scope"]
        n = max(1, len(gen))
        cite_checks = [citation_hit(r["expected_pages"], r[name].get("citations", [])) for r in rows if "error" not in r[name]]
        summary_rows.append([
            gemini.config.llm_model_id if name == "gemini" else local.config.llm_model_id,
            f"{len(ok)}/{len(rows)}",
            round(sum(r["llm_sec"] for r in gen) / n, 2),
            round(sum(r["total_sec"] for r in gen) / n, 2),
            round(sum(r["tokens_per_sec"] for r in gen) / n, 1),
            round(sum(r["output_tokens"] for r in gen) / n, 1),
            f"{sum(r['grounding_status'] == 'grounded' for r in gen)}/{len(gen)}",
            f"{cite_checks.count('yes')}/{cite_checks.count('yes') + cite_checks.count('no')}",
            round(sum(r["cost_usd"] for r in ok), 6)
        ])
    headers = ["Model", "Succeeded", "Avg LLM (s)", "Avg Total (s)", "Avg tok/s", "Avg Out Tokens", "Grounded", "Cites Expected Page", "Total Cost ($)"]
    summary = tabulate(summary_rows, headers=headers, tablefmt="github")
    print("\n" + summary)

    os.makedirs(local.config.report_dir, exist_ok=True)
    # One report per (document, GGUF) so different quantizations don't overwrite each other
    tag = f"{os.path.splitext(os.path.basename(config.pdf_path))[0]}_{os.path.splitext(os.path.basename(served))[0]}"
    json_path = os.path.join(local.config.report_dir, f"gemini_vs_local_{tag}.json")
    md_path = os.path.join(local.config.report_dir, f"gemini_vs_local_{tag}.md")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"gemini_model": gemini.config.llm_model_id, "local_model": local.config.llm_model_id,
                   "local_gguf": served, "pdf_path": config.pdf_path, "rows": rows}, f, indent=2)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Gemini vs Local LLM\n\nLocal GGUF: `{served}`\n\n{summary}\n\n")
        for r in rows:
            f.write(f"## {r['question']}\n\nExpected pages: {r['expected_pages']}\n\n")
            for name in ("gemini", "local"):
                res = r[name]
                body = res.get("error") or f"{res['answer']}\n\n_cites {res['citations']}, {res['grounding_status']}, LLM {res['llm_sec']}s_"
                f.write(f"**{name}**: {body}\n\n")
    print(f"\n[REPORT] {json_path}\n[REPORT] {md_path}")


if __name__ == "__main__":
    main()
