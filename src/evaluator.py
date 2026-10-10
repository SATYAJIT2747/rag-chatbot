import os
from typing import List, Dict, Any, Optional
from langchain_community.vectorstores import FAISS
from .indexer import LocalBM25Retriever
from .models import ModelRegistry
from eval.gold import load_gold_items, item_sources, is_answerable

# Questions, reference answers and evidence pages live in data/gold_dataset.json
try:
    DEFAULT_EVAL_SUITE = load_gold_items()
except FileNotFoundError as e:
    print(f"[Evaluator] Warning: gold dataset not found - {e}")
    DEFAULT_EVAL_SUITE = []

def gold_suite_for(pdf_path: str, eval_suite: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """Gold questions answerable from this one PDF, plus the unanswerable (refusal) questions."""
    source = os.path.basename(pdf_path)
    suite = DEFAULT_EVAL_SUITE if eval_suite is None else eval_suite
    return [item for item in suite if not is_answerable(item) or item_sources(item) == {source}]

def evaluate_retrieval_suite(
    eval_suite: List[Dict[str, Any]],
    vectorstore: FAISS,
    bm25_ret: LocalBM25Retriever,
    registry: ModelRegistry
) -> Dict[str, Any]:
    """Scores each retrieval method on the gold questions whose evidence documents are in this index."""
    from eval.retrieval_ablation import run_ablation
    return run_ablation(eval_suite, vectorstore, bm25_ret, registry)
