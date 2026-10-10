"""
Session-memory evaluation over the follow-up questions in the gold dataset
(the "conversations" list in data/gold_dataset.json).

Each follow-up is retrieved with hybrid + reranker three ways:
  - memory off: the follow-up exactly as typed
  - memory on:  the follow-up rewritten by src.memory.condense_question, per model
  - oracle:     the hand-written standalone question (upper bound)
Earlier turns are built from the gold question and expected answer of history_ids.

Usage (from pdf_rag_bot/):
    ..\\.venv\\Scripts\\python.exe -m eval.memory_eval
    ..\\.venv\\Scripts\\python.exe -m eval.memory_eval --models Qwen3.5-4B-Base-Q6_K gemini-flash-lite-latest
"""
import os
import sys
import json
import time
import argparse
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import RAGConfig, get_system_device_info
from src.models import ModelRegistry
from src.indexer import LocalBM25Retriever, chunk_documents, build_embeddings, build_faiss
from src.retrieval import hybrid_retrieve, rerank_documents
from src.memory import condense_question
from eval.gold import DEFAULT_GOLD_PATH, load_gold, gold_pages
from eval.model_comparison import registry_for
from eval.retrieval_ablation import load_corpus, index_dir_for
from eval.retrieval_metrics import first_hit_rank, hit_at_k, mrr_at_k, mean

DEFAULT_MODELS = ["gemini-flash-lite-latest"]


def history_turns(conversation: Dict[str, Any], items_by_id: Dict[str, Dict[str, Any]]) -> List[Dict[str, str]]:
    return [{"question": items_by_id[i]["question"], "answer": items_by_id[i]["expected_answer"]}
            for i in conversation["history_ids"]]


def summarize(ranks: List[Any]) -> Dict[str, Any]:
    retrieved = [[False] * (r - 1) + [True] if r else [] for r in ranks]
    gold = {True}
    return {
        "Hit@1": round(mean([hit_at_k(r, gold, 1) for r in retrieved]), 3),
        "Hit@5": round(mean([hit_at_k(r, gold, 5) for r in retrieved]), 3),
        "MRR@10": round(mean([mrr_at_k(r, gold, 10) for r in retrieved]), 3)
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate session memory on gold follow-up questions")
    parser.add_argument("--gold", type=str, default=DEFAULT_GOLD_PATH, help="Path to the gold dataset JSON")
    parser.add_argument("--models", type=str, nargs="+", default=DEFAULT_MODELS,
                        help="LLMs that rewrite the follow-ups: llama-server model ids or Gemini model names")
    args = parser.parse_args()

    gold_path = os.path.abspath(args.gold)
    os.chdir(PROJECT_ROOT)
    from tabulate import tabulate

    gold = load_gold(gold_path)
    conversations = gold.get("conversations", [])
    items_by_id = {item["id"]: item for item in gold["items"]}
    pdf_paths = [os.path.join(os.path.dirname(gold_path), name) for name in gold["documents"]]

    config = RAGConfig()
    device = get_system_device_info(config)["device"]
    base_registry = ModelRegistry(config, device=device)

    pages = load_corpus(pdf_paths)
    chunks = chunk_documents(pages, chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
    embeddings = build_embeddings(config.embedding_model_id, device=device)
    vectorstore = build_faiss(chunks, embeddings, index_dir_for(chunks, config, config.chunk_size))
    bm25_ret = LocalBM25Retriever(chunks)

    def gold_rank(query: str, conversation: Dict[str, Any]) -> Any:
        """Rank of the first gold page for this query through hybrid + reranker (None if not in the top 20)."""
        candidates = hybrid_retrieve(query, vectorstore, bm25_ret, top_k=config.retrieval_k, rrf_k=config.rrf_k)["candidates"]
        head = rerank_documents(query, candidates[:config.rerank_k], base_registry, top_k=config.rerank_k)["reranked_docs"]
        retrieved = [(doc.metadata.get("source"), doc.metadata["page"]) for doc in head + candidates[config.rerank_k:]]
        return first_hit_rank(retrieved, gold_pages(conversation), config.retrieval_k)

    rows = [{"id": c["id"], "kind": c["kind"], "follow_up": c["follow_up"], "expected_standalone": c["expected_standalone"],
             "rank": {"Memory off": gold_rank(c["follow_up"], c), "Oracle rewrite": gold_rank(c["expected_standalone"], c)},
             "rewrites": {}}
            for c in conversations]

    rewrite_ms: Dict[str, float] = {}
    skipped: Dict[str, str] = {}
    for model in args.models:
        print(f"\n[eval] Rewriting follow-ups with {model}")
        label = f"Memory on ({model})"
        try:
            registry, limiter = registry_for(model, base_registry, config)
            latencies = []
            for conversation, row in zip(conversations, rows):
                if limiter:
                    limiter.wait()
                t0 = time.time()
                standalone = condense_question(conversation["follow_up"], history_turns(conversation, items_by_id), registry)
                latencies.append((time.time() - t0) * 1000.0)
                row["rewrites"][model] = standalone
                row["rank"][label] = gold_rank(standalone, conversation)
                print(f"  {row['id']}: {standalone}")
            rewrite_ms[label] = round(mean(latencies), 1)
        except Exception as e:
            skipped[model] = f"{type(e).__name__}: {e}"
            print(f"[eval] Skipping {model}: {skipped[model]}")
            for row in rows:
                row["rank"].pop(label, None)
                row["rewrites"].pop(model, None)

    conditions = ["Memory off"] + [f"Memory on ({m})" for m in args.models if m not in skipped] + ["Oracle rewrite"]
    summary = {}
    for kind, title in (("follow_up", "Follow-ups"), ("control", "Already-standalone controls")):
        selected = [r for r in rows if r["kind"] == kind]
        summary[title] = {"count": len(selected),
                          "conditions": {c: summarize([r["rank"][c] for r in selected]) for c in conditions}}

    def summary_table(title: str) -> str:
        block = summary[title]["conditions"]
        table_rows = [[c, block[c]["Hit@1"], block[c]["Hit@5"], block[c]["MRR@10"], rewrite_ms.get(c, "-")] for c in conditions]
        return tabulate(table_rows, headers=["Condition", "Hit@1", "Hit@5", "MRR@10", "Rewrite latency (ms)"], tablefmt="github")

    detail = tabulate(
        [[r["id"], r["follow_up"][:60]] + [r["rank"][c] or "-" for c in conditions] for r in rows],
        headers=["ID", "Follow-up"] + conditions, tablefmt="github"
    )
    for title in summary:
        print(f"\n{title} ({summary[title]['count']} questions)\n{summary_table(title)}")

    report_dir = os.path.join("artifacts", "eval", "report")
    os.makedirs(report_dir, exist_ok=True)
    json_path = os.path.join(report_dir, "memory_eval.json")
    md_path = os.path.join(report_dir, "memory_eval.md")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"models": args.models, "skipped_models": skipped, "summary": summary,
                   "rewrite_latency_ms": rewrite_ms, "rows": rows}, f, indent=2, ensure_ascii=False)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Session Memory Evaluation\n\n")
        f.write(f"Retrieval is hybrid + reranker at chunk size {config.chunk_size}. A hit is a gold (document, page) "
                "among the retrieved chunks.\n\n")
        for title in summary:
            f.write(f"## {title} ({summary[title]['count']} questions)\n\n{summary_table(title)}\n\n")
        f.write(f"## Rank of the first gold page (`-` = not in the top {config.retrieval_k})\n\n{detail}\n\n")
        f.write("## Rewrites\n\n")
        for r in rows:
            f.write(f"- `{r['id']}`: {r['follow_up']}\n")
            for model, standalone in r["rewrites"].items():
                f.write(f"  - {model}: {standalone}\n")
        for model, reason in skipped.items():
            f.write(f"\nModel skipped: **{model}** ({reason})\n")
    print(f"\n[REPORT] {json_path}\n[REPORT] {md_path}")


if __name__ == "__main__":
    main()
