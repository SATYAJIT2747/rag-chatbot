"""
Retrieval ablation over the gold dataset (data/gold_dataset.json).

Indexes every PDF listed in the gold dataset into one combined index, runs each
retrieval method over the answerable questions and scores it against the gold
(source, page) evidence. No answer generation is involved.

Usage (from pdf_rag_bot/):
    ..\\.venv\\Scripts\\python.exe -m eval.retrieval_ablation
    ..\\.venv\\Scripts\\python.exe -m eval.retrieval_ablation --chunk-sizes 256 512 1024
    ..\\.venv\\Scripts\\python.exe -m eval.retrieval_ablation --rewrite local
"""
import os
import sys
import json
import time
import hashlib
import argparse
from typing import Any, Callable, Dict, List, Optional, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from src.config import RAGConfig, get_system_device_info
from src.models import ModelRegistry
from src.document_loader import load_and_preprocess_pdf
from src.indexer import LocalBM25Retriever, chunk_documents, build_embeddings, build_faiss
from src.retrieval import dense_retrieve, bm25_retrieve, hybrid_retrieve, rerank_documents, rewrite_query
from eval.gold import DEFAULT_GOLD_PATH, load_gold, gold_pages, item_sources, is_answerable
from eval.rate_limit import RateLimiter
from eval.retrieval_metrics import first_hit_rank, hit_at_k, recall_at_k, mrr_at_k, ndcg_at_k, mean

REWRITE_METHOD = "Query Rewrite + Hybrid"
METRIC_COLUMNS = ["Hit@1", "Hit@5", "Recall@5", "Recall@10", "Recall@20", "MRR@10", "nDCG@10", "Latency (ms)"]


def select_items(eval_suite: List[Dict[str, Any]], chunks: List[Document]) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    """Keeps the answerable questions whose evidence documents are indexed with real text."""
    indexed = {c.metadata.get("source") for c in chunks}
    with_text = {c.metadata.get("source") for c in chunks if c.metadata.get("content_type") != "ocr_skipped"}

    items, skipped = [], []
    for item in eval_suite:
        if not is_answerable(item):
            continue
        sources = item_sources(item)
        if not sources <= indexed:
            reason = "evidence document is not in the index"
        elif not sources <= with_text:
            reason = "evidence document has no extractable text (needs OCR)"
        else:
            items.append(item)
            continue
        skipped.append({"id": item.get("id", ""), "question": item["question"], "reason": reason})
    return items, skipped


def run_ablation(
    eval_suite: List[Dict[str, Any]],
    vectorstore: FAISS,
    bm25_ret: LocalBM25Retriever,
    registry: ModelRegistry,
    retrieval_k: int = 20,
    rerank_k: int = 10,
    rrf_k: int = 60,
    include_rewrite: bool = True,
    rewrite_limiter: Optional[RateLimiter] = None
) -> Dict[str, Any]:
    items, skipped = select_items(eval_suite, bm25_ret.chunks)
    result = {"methods": {}, "total_queries": len(items), "skipped": skipped, "skipped_methods": {}, "per_question": []}
    if not items:
        return result

    if include_rewrite and rewrite_limiter is None and "gemini" in registry.config.llm_model_id.lower():
        rewrite_limiter = RateLimiter.for_gemini()

    def bm25(q: str) -> List[Document]:
        return [doc for doc, _, _ in bm25_retrieve(q, bm25_ret, top_k=retrieval_k)]

    def dense(q: str) -> List[Document]:
        return [doc for doc, _, _ in dense_retrieve(q, vectorstore, top_k=retrieval_k)]

    def hybrid(q: str) -> List[Document]:
        return hybrid_retrieve(q, vectorstore, bm25_ret, top_k=retrieval_k, rrf_k=rrf_k)["candidates"]

    def hybrid_rerank(q: str) -> List[Document]:
        # Same as the pipeline: only the first rerank_k candidates go through the cross-encoder
        candidates = hybrid(q)
        head = rerank_documents(q, candidates[:rerank_k], registry, top_k=rerank_k)["reranked_docs"]
        return head + candidates[rerank_k:]

    def rewrite_hybrid(q: str) -> List[Document]:
        return hybrid(rewrite_query(q, registry) or q)

    methods: Dict[str, Callable[[str], List[Document]]] = {
        "BM25": bm25,
        "Dense": dense,
        "Hybrid + RRF": hybrid,
        "Hybrid + Reranker": hybrid_rerank
    }
    if include_rewrite:
        methods[REWRITE_METHOD] = rewrite_hybrid

    # Load the embedding and reranker models before anything is timed
    hybrid_rerank("warm up")

    per_question = [
        {"id": item.get("id", ""), "question": item["question"], "sources": sorted(item_sources(item)), "first_hit_rank": {}}
        for item in items
    ]

    for name, retrieve in methods.items():
        scores: Dict[str, List[float]] = {col: [] for col in METRIC_COLUMNS}
        ranks = []
        try:
            for item in items:
                gold = gold_pages(item)
                if name == REWRITE_METHOD and rewrite_limiter:
                    rewrite_limiter.wait()
                t0 = time.time()
                docs = retrieve(item["question"])
                latency_ms = (time.time() - t0) * 1000.0

                # Legacy items carry page numbers only, so the source is ignored for them
                legacy = "evidence" not in item
                retrieved = [(None if legacy else doc.metadata.get("source"), doc.metadata["page"]) for doc in docs]

                scores["Hit@1"].append(hit_at_k(retrieved, gold, 1))
                scores["Hit@5"].append(hit_at_k(retrieved, gold, 5))
                scores["Recall@5"].append(recall_at_k(retrieved, gold, 5))
                scores["Recall@10"].append(recall_at_k(retrieved, gold, 10))
                scores["Recall@20"].append(recall_at_k(retrieved, gold, 20))
                scores["MRR@10"].append(mrr_at_k(retrieved, gold, 10))
                scores["nDCG@10"].append(ndcg_at_k(retrieved, gold, 10))
                scores["Latency (ms)"].append(latency_ms)
                ranks.append(first_hit_rank(retrieved, gold, retrieval_k))
        except Exception as e:
            # Usually the rewrite LLM being unreachable or out of quota; the other methods still stand
            result["skipped_methods"][name] = f"{type(e).__name__}: {e}"
            print(f"[eval] Skipping '{name}': {type(e).__name__}: {e}")
            continue

        result["methods"][name] = {
            col: round(mean(vals), 1 if col == "Latency (ms)" else 3) for col, vals in scores.items()
        }
        for record, rank in zip(per_question, ranks):
            record["first_hit_rank"][name] = rank

    result["per_question"] = per_question
    return result


def load_corpus(pdf_paths: List[str]) -> List[Document]:
    pages = []
    for path in pdf_paths:
        # load_and_preprocess_pdf would silently generate a demo PDF for a missing path
        if not os.path.exists(path):
            raise FileNotFoundError(f"PDF listed in the gold dataset was not found: {path}")
        docs, _ = load_and_preprocess_pdf(path)
        pages.extend(docs)
    return pages


def index_dir_for(chunks: List[Document], config: RAGConfig, chunk_size: int) -> str:
    # src/indexer reuses whatever index is on disk, so the directory is keyed by the exact chunks it holds
    hasher = hashlib.sha1(config.embedding_model_id.encode("utf-8"))
    for chunk in chunks:
        hasher.update(chunk.metadata["chunk_id"].encode("utf-8"))
        hasher.update(chunk.page_content.encode("utf-8"))
    return os.path.join("artifacts", "eval", "index", f"cs{chunk_size}_{hasher.hexdigest()[:12]}")


def methods_table(run: Dict[str, Any]) -> str:
    from tabulate import tabulate
    rows = [[name] + [metrics[col] for col in METRIC_COLUMNS] for name, metrics in run["methods"].items()]
    return tabulate(rows, headers=["Method"] + METRIC_COLUMNS, tablefmt="github")


def per_document_table(run: Dict[str, Any]) -> str:
    from tabulate import tabulate
    methods = list(run["methods"].keys())
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for record in run["per_question"]:
        sources = record["sources"]
        label = sources[0] if len(sources) == 1 else ("cross-document" if sources else "unspecified")
        groups.setdefault(label, []).append(record)

    rows = []
    for label, records in groups.items():
        row = [label, len(records)]
        for m in methods:
            hits = sum(1 for r in records if r["first_hit_rank"].get(m) and r["first_hit_rank"][m] <= 5)
            row.append(f"{hits}/{len(records)}")
        rows.append(row)
    return tabulate(rows, headers=["Document", "Questions"] + methods, tablefmt="github")


def per_question_table(run: Dict[str, Any]) -> str:
    from tabulate import tabulate
    methods = list(run["methods"].keys())
    rows = [
        [r["id"], r["question"][:70]] + [r["first_hit_rank"].get(m) or "-" for m in methods]
        for r in run["per_question"]
    ]
    return tabulate(rows, headers=["ID", "Question"] + methods, tablefmt="github")


def write_report(report: Dict[str, Any], report_dir: str) -> Tuple[str, str]:
    os.makedirs(report_dir, exist_ok=True)
    json_path = os.path.join(report_dir, "retrieval_ablation.json")
    md_path = os.path.join(report_dir, "retrieval_ablation.md")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    cfg = report["configuration"]
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Retrieval Ablation\n\n")
        f.write(f"- Documents: {', '.join(cfg['documents'])}\n")
        f.write(f"- Embeddings: `{cfg['embedding_model']}`, reranker: `{cfg['reranker_model']}`\n")
        f.write(f"- retrieval_k={cfg['retrieval_k']}, rerank_k={cfg['rerank_k']}, rrf_k={cfg['rrf_k']}, "
                f"chunk_overlap={cfg['chunk_overlap']}\n")
        f.write(f"- Rewrite LLM: {cfg['rewrite_llm']}\n\n")
        f.write("A hit is a retrieved chunk whose (document, page) is in the gold evidence. "
                "Hit@k is 1 if any gold page is in the top k chunks; Recall@k is the share of gold pages found.\n\n")
        for run in report["runs"]:
            f.write(f"## Chunk size {run['chunk_size']} ({run['num_chunks']} chunks, {run['total_queries']} questions)\n\n")
            if not run["methods"]:
                f.write("No questions could be scored.\n\n")
                continue
            f.write(methods_table(run) + "\n\n")
            f.write("### Hit@5 by document\n\n" + per_document_table(run) + "\n\n")
            f.write("### Rank of the first gold page per question (`-` = not in the top "
                    f"{cfg['retrieval_k']})\n\n" + per_question_table(run) + "\n\n")
            for name, reason in run["skipped_methods"].items():
                f.write(f"Method skipped: **{name}** ({reason})\n\n")
        skipped = report["runs"][0]["skipped"] if report["runs"] else []
        if skipped:
            f.write(f"## Questions not scored ({len(skipped)})\n\n")
            for s in skipped:
                f.write(f"- `{s['id']}`: {s['reason']}\n")
    return json_path, md_path


def main():
    parser = argparse.ArgumentParser(description="Retrieval ablation over the gold dataset")
    parser.add_argument("--gold", type=str, default=DEFAULT_GOLD_PATH, help="Path to the gold dataset JSON")
    parser.add_argument("--chunk-sizes", type=int, nargs="+", default=None,
                        help="Chunk sizes to compare (default: RAGConfig.chunk_size only)")
    parser.add_argument("--rewrite", choices=["none", "local", "gemini"], default="none",
                        help="LLM used for the query-rewrite method (default: none, method is left out)")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild of the eval FAISS indexes")
    args = parser.parse_args()

    gold_path = os.path.abspath(args.gold)
    os.chdir(PROJECT_ROOT)

    gold = load_gold(gold_path)
    pdf_paths = [os.path.join(os.path.dirname(gold_path), name) for name in gold["documents"]]

    config = RAGConfig()
    device = get_system_device_info(config)["device"]
    registry = ModelRegistry(config, device=device)
    if args.rewrite == "local":
        from local_llm.llm_client import client_from_env
        config.llm_model_id = os.getenv("LOCAL_LLM_MODEL", "qwen3.5-4b-q8")
        registry.llm_model = client_from_env(temperature=config.llm_temperature, max_tokens=config.llm_max_new_tokens)

    print(f"[eval] Loading {len(pdf_paths)} PDFs from {os.path.dirname(gold_path)}")
    pages = load_corpus(pdf_paths)
    embeddings = build_embeddings(config.embedding_model_id, device=device)

    runs = []
    for chunk_size in args.chunk_sizes or [config.chunk_size]:
        chunks = chunk_documents(pages, chunk_size=chunk_size, chunk_overlap=config.chunk_overlap)
        index_dir = index_dir_for(chunks, config, chunk_size)
        vectorstore = build_faiss(chunks, embeddings, index_dir, force_rebuild=args.rebuild)
        bm25_ret = LocalBM25Retriever(chunks)

        print(f"\n[eval] Chunk size {chunk_size}: {len(chunks)} chunks")
        run = run_ablation(
            gold["items"], vectorstore, bm25_ret, registry,
            retrieval_k=config.retrieval_k, rerank_k=config.rerank_k, rrf_k=config.rrf_k,
            include_rewrite=args.rewrite != "none"
        )
        runs.append({"chunk_size": chunk_size, "num_chunks": len(chunks), **run})
        if run["methods"]:
            print(methods_table(run))

    for s in runs[0]["skipped"]:
        print(f"[eval] Not scored: {s['id']} ({s['reason']})")

    report = {
        "configuration": {
            "gold_path": gold_path,
            "documents": gold["documents"],
            "embedding_model": config.embedding_model_id,
            "reranker_model": config.reranker_model_id,
            "rewrite_llm": config.llm_model_id if args.rewrite != "none" else "none",
            "retrieval_k": config.retrieval_k,
            "rerank_k": config.rerank_k,
            "rrf_k": config.rrf_k,
            "chunk_overlap": config.chunk_overlap,
            "device": device
        },
        "runs": runs
    }
    json_path, md_path = write_report(report, os.path.join("artifacts", "eval", "report"))
    print(f"\n[REPORT] {json_path}\n[REPORT] {md_path}")


if __name__ == "__main__":
    main()
