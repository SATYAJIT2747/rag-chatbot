"""
Answer-model comparison over the gold dataset (data/gold_dataset.json).

Retrieval is fixed (hybrid + reranker over one combined index of every gold PDF)
and done once per question, so every model answers from the exact same context.
Guardrails run as in the pipeline; the response cache is not used.

Model names containing "gemini" go through the Gemini API (throttled to GEMINI_RPM,
default 10 calls/min); any other name is a model id served by llama-server.

Usage (from pdf_rag_bot/):
    ..\\.venv\\Scripts\\python.exe -m eval.model_comparison
    ..\\.venv\\Scripts\\python.exe -m eval.model_comparison --models Qwen3.5-4B-Base-Q8_0 gemini-flash-lite-latest
"""
import os
import re
import sys
import copy
import json
import math
import argparse
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from langchain_core.documents import Document
from src.config import RAGConfig, get_system_device_info
from src.models import ModelRegistry
from src.indexer import LocalBM25Retriever, chunk_documents, build_embeddings, build_faiss
from src.retrieval import hybrid_retrieve, rerank_documents
from src.generation import generate_answer, extract_citations
from src.guardrails import RAGGuardrailsManager
from src.metrics import calculate_llm_cost
from local_llm.llm_client import LlamaServerChat
from eval.gold import DEFAULT_GOLD_PATH, load_gold, gold_pages, is_answerable
from eval.rate_limit import RateLimiter
from eval.retrieval_ablation import select_items, load_corpus, index_dir_for
from eval.retrieval_metrics import mean

DEFAULT_MODELS = ["Qwen3.5-4B-Base-Q6_K", "Qwen3.5-4B-Base-Q8_0"]
REFUSAL_ANSWER = "The provided document does not contain enough information to answer this."
# Models often refuse in their own words, so this is wider than the exact refusal sentence
REFUSAL_PATTERN = re.compile(
    r"(does not|doesn't|do not) contain (enough |any |sufficient )?information"
    r"|there is no information|no information (about|on|regarding|explaining)"
    r"|i can only answer questions related to the uploaded pdf"
)
NUMERIC_FACT = re.compile(r"[\d.]+")


def build_context(
    chunks: List[Document],
    page_map: Dict[Tuple[str, int], str],
    return_parent_context: bool,
    max_context_chars: int
) -> Tuple[str, List[Tuple[str, int]]]:
    """src.generation.build_context, but keyed by (source, page) so pages of different PDFs don't collide."""
    blocks, pages, total_chars = [], [], 0
    for chunk in chunks:
        key = (chunk.metadata["source"], chunk.metadata["page"])
        if return_parent_context:
            if key in pages:
                continue
            block = f"--- [Page {key[1]}] (Source: {key[0]}) ---\n{page_map.get(key, chunk.page_content)}\n"
        else:
            block = f"--- [Page {key[1]}] (Chunk: {chunk.metadata.get('chunk_id', 'unknown')}) ---\n{chunk.page_content}\n"
        if total_chars + len(block) > max_context_chars and blocks:
            break
        blocks.append(block)
        pages.append(key)
        total_chars += len(block)
    return "\n".join(blocks), pages


def prepare_question(
    question: str,
    vectorstore: Any,
    bm25_ret: LocalBM25Retriever,
    registry: ModelRegistry,
    config: RAGConfig,
    guardrails: RAGGuardrailsManager,
    page_map: Dict[Tuple[str, int], str]
) -> Dict[str, Any]:
    """Everything up to the LLM call: input guard, retrieval, rerank, scope guard, context."""
    ok, message, status = guardrails.check_input(question)
    if not ok:
        return {"blocked": status, "answer": message, "context": "", "context_pages": []}

    candidates = hybrid_retrieve(question, vectorstore, bm25_ret, top_k=config.retrieval_k, rrf_k=config.rrf_k)["candidates"]
    reranked = rerank_documents(question, candidates[:config.rerank_k], registry, top_k=config.final_k)
    final_docs = reranked["reranked_docs"]

    in_scope, scope_message = guardrails.check_scope(final_docs, {"rerank_debug": reranked["rerank_debug"]})
    if not in_scope:
        return {"blocked": "blocked_scope", "answer": scope_message, "context": "", "context_pages": []}

    context, pages = build_context(final_docs, page_map, config.return_parent_context, config.max_parent_context_chars)
    return {"blocked": None, "answer": None, "context": context, "context_pages": pages}


def registry_for(model_name: str, base_registry: ModelRegistry, config: RAGConfig) -> Tuple[ModelRegistry, Optional[RateLimiter]]:
    """Shares the embedding/reranker models of base_registry; only the LLM differs."""
    cfg = copy.deepcopy(config)
    cfg.llm_model_id = model_name
    registry = copy.copy(base_registry)
    registry.config = cfg
    registry.llm_model = None
    registry.llm_tokenizer = None

    if "gemini" in model_name.lower():
        registry.load_llm()
        return registry, RateLimiter.for_gemini()

    cfg.model_pricing[model_name] = {"input_per_1m": 0.00, "output_per_1m": 0.00}
    registry.llm_model = LlamaServerChat(
        base_url=os.getenv("LOCAL_LLM_BASE_URL", "http://127.0.0.1:8080/v1"),
        model=model_name,
        temperature=cfg.llm_temperature,
        max_tokens=cfg.llm_max_new_tokens,
        enable_thinking=os.getenv("LOCAL_LLM_THINKING", "false").lower() == "true"
    )
    # Makes llama-server load (or swap to) this model before any question is timed
    registry.llm_model.invoke([("user", "Say OK.")])
    return registry, None


def answer_question(
    question: str,
    prepared: Dict[str, Any],
    registry: ModelRegistry,
    guardrails: RAGGuardrailsManager,
    limiter: Optional[RateLimiter]
) -> Dict[str, Any]:
    if prepared["blocked"]:
        return {"answer": prepared["answer"], "guardrail_status": prepared["blocked"], "generated": False}

    if limiter:
        limiter.wait()
    cfg = registry.config
    try:
        gen = generate_answer(question, prepared["context"], registry, cfg.llm_max_new_tokens, cfg.llm_temperature)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "generated": False}

    grounded = guardrails.check_output_grounding(question, prepared["context"], gen["answer"])["grounded"]
    llm_sec = gen["latency_sec"]
    return {
        "answer": gen["answer"] if grounded else REFUSAL_ANSWER,
        "raw_answer": gen["answer"],
        "guardrail_status": "allowed" if grounded else "blocked_ungrounded",
        "generated": True,
        "grounded": grounded,
        "llm_sec": llm_sec,
        "input_tokens": gen["input_tokens"],
        "output_tokens": gen["output_tokens"],
        "tokens_per_sec": round(gen["output_tokens"] / llm_sec, 1) if llm_sec else 0.0,
        "cost_usd": calculate_llm_cost(cfg.llm_model_id, gen["input_tokens"], gen["output_tokens"], cfg.model_pricing)
    }


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    for dash in ("−", "–", "—"):
        text = text.replace(dash, "-")
    return re.sub(r"\s+", " ", text.replace(",", ""))


def key_fact_recall(answer: str, key_facts: List[str]) -> Optional[float]:
    """Share of key facts present in the answer. Each fact is '|'-separated alternatives."""
    if not key_facts:
        return None
    text = normalize(answer)

    def present(alternative: str) -> bool:
        alt = normalize(alternative)
        if NUMERIC_FACT.fullmatch(alt):
            # A bare number must not match inside a longer one ("8" in "248")
            return re.search(rf"(?<![\d.]){re.escape(alt)}(?!\d)", text) is not None
        return alt in text

    return sum(any(present(a) for a in fact.split("|")) for fact in key_facts) / len(key_facts)


def is_refusal(answer: str) -> bool:
    # Only the opening counts: a partial answer that ends with "the document does not contain ..." is not a refusal
    return REFUSAL_PATTERN.search(answer.replace("*", "").lower()[:200]) is not None


def score_answer(item: Dict[str, Any], result: Dict[str, Any], prepared: Dict[str, Any], embedder: Any) -> Dict[str, Any]:
    if "error" in result:
        return {}
    answer = result["answer"]
    scores: Dict[str, Any] = {"refused": is_refusal(answer)}
    if is_answerable(item):
        gold = gold_pages(item)
        scores["key_fact_recall"] = key_fact_recall(answer, item.get("key_facts", []))
        # What the model wrote before the grounding guard could replace it with a refusal
        scores["raw_key_fact_recall"] = key_fact_recall(result.get("raw_answer", answer), item.get("key_facts", []))
        vectors = embedder.encode([answer, item["expected_answer"]], normalize_embeddings=True)
        scores["semantic_similarity"] = float(vectors[0] @ vectors[1])
        scores["cites_gold_page"] = bool(set(extract_citations(answer)) & {page for _, page in gold})
        scores["gold_page_in_context"] = bool(gold & set(prepared["context_pages"]))
    return scores


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(pct / 100.0 * len(ordered)) - 1)]


def rate(flags: List[bool]) -> str:
    return f"{sum(flags)}/{len(flags)} ({100.0 * sum(flags) / len(flags):.0f}%)" if flags else "n/a"


def summarize(rows: List[Dict[str, Any]], model: str) -> Dict[str, Any]:
    ok = [r for r in rows if "error" not in r[model]]
    answerable = [r for r in ok if r["answerable"]]
    unanswerable = [r for r in ok if not r["answerable"]]
    generated = [r[model] for r in ok if r[model]["generated"]]
    recalls = [r[model]["scores"]["key_fact_recall"] for r in answerable if r[model]["scores"]["key_fact_recall"] is not None]
    raw_recalls = [r[model]["scores"]["raw_key_fact_recall"] for r in answerable if r[model]["scores"]["raw_key_fact_recall"] is not None]
    wrongly_blocked = [r[model]["guardrail_status"] == "blocked_ungrounded" and r[model]["scores"]["raw_key_fact_recall"] == 1.0
                       for r in answerable]
    return {
        "Key-fact recall": round(mean(recalls), 3),
        "All key facts present": rate([v == 1.0 for v in recalls]),
        "Key-fact recall before grounding guard": round(mean(raw_recalls), 3),
        "Correct answers blocked by grounding guard": rate(wrongly_blocked),
        "Semantic similarity": round(mean([r[model]["scores"]["semantic_similarity"] for r in answerable]), 3),
        "Cites a gold page": rate([r[model]["scores"]["cites_gold_page"] for r in answerable]),
        "False refusal (answerable)": rate([r[model]["scores"]["refused"] for r in answerable]),
        "Correct refusal (unanswerable)": rate([r[model]["scores"]["refused"] for r in unanswerable]),
        "Passed grounding guard": rate([g["grounded"] for g in generated]),
        "Avg LLM latency (s)": round(mean([g["llm_sec"] for g in generated]), 2),
        "p95 LLM latency (s)": round(percentile([g["llm_sec"] for g in generated], 95), 2),
        "Avg tokens/s": round(mean([g["tokens_per_sec"] for g in generated]), 1),
        "Avg output tokens": round(mean([g["output_tokens"] for g in generated]), 1),
        "Total cost ($)": round(sum(g["cost_usd"] for g in generated), 6),
        "Errors": len(rows) - len(ok)
    }


def summary_table(summaries: Dict[str, Dict[str, Any]]) -> str:
    from tabulate import tabulate
    models = list(summaries.keys())
    metrics = list(summaries[models[0]].keys())
    rows = [[metric] + [summaries[m][metric] for m in models] for metric in metrics]
    return tabulate(rows, headers=["Metric"] + models, tablefmt="github")


def write_report(report: Dict[str, Any], report_dir: str) -> Tuple[str, str]:
    os.makedirs(report_dir, exist_ok=True)
    json_path = os.path.join(report_dir, "model_comparison.json")
    md_path = os.path.join(report_dir, "model_comparison.md")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    cfg = report["configuration"]
    models = list(report["summary"].keys())
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Model Comparison\n\n")
        f.write(f"- Documents: {', '.join(cfg['documents'])}\n")
        f.write(f"- Retrieval (same for every model): hybrid + reranker, chunk_size={cfg['chunk_size']}, final_k={cfg['final_k']}, "
                f"max context {cfg['max_context_chars']} chars\n")
        f.write(f"- Questions: {cfg['answerable']} answerable, {cfg['unanswerable']} unanswerable\n")
        f.write(f"- Gold page present in the context: {cfg['gold_page_in_context']}\n\n")
        f.write(summary_table(report["summary"]) + "\n\n")
        f.write("Key-fact recall is the share of the gold `key_facts` found in the answer. "
                "Latency, tokens/s and the grounding guard cover only questions that reached the LLM.\n\n")
        if report["skipped"]:
            f.write(f"## Questions not scored ({len(report['skipped'])})\n\n")
            for s in report["skipped"]:
                f.write(f"- `{s['id']}`: {s['reason']}\n")
            f.write("\n")
        f.write("## Answers\n\n")
        for row in report["rows"]:
            f.write(f"### {row['id']}: {row['question']}\n\n**Expected**: {row['expected_answer']}\n\n")
            for m in models:
                res = row[m]
                if "error" in res:
                    f.write(f"**{m}**: ERROR {res['error']}\n\n")
                    continue
                note = f"key facts {res['scores']['key_fact_recall']}, " if row["answerable"] else ""
                f.write(f"**{m}**: {res['answer']}\n\n_{note}{res['guardrail_status']}, LLM {res.get('llm_sec', 0.0)}s_\n\n")
    return json_path, md_path


def main():
    parser = argparse.ArgumentParser(description="Compare answer models on the gold dataset")
    parser.add_argument("--gold", type=str, default=DEFAULT_GOLD_PATH, help="Path to the gold dataset JSON")
    parser.add_argument("--models", type=str, nargs="+", default=DEFAULT_MODELS,
                        help="llama-server model ids, or Gemini model names (anything containing 'gemini')")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N questions (smoke test)")
    parser.add_argument("--rescore", action="store_true",
                        help="Re-score the answers saved in the last report without calling any model")
    args = parser.parse_args()

    gold_path = os.path.abspath(args.gold)
    os.chdir(PROJECT_ROOT)
    report_dir = os.path.join("artifacts", "eval", "report")

    gold = load_gold(gold_path)
    if args.rescore:
        rescore(gold, report_dir)
        return
    pdf_paths = [os.path.join(os.path.dirname(gold_path), name) for name in gold["documents"]]

    config = RAGConfig()
    device = get_system_device_info(config)["device"]
    base_registry = ModelRegistry(config, device=device)
    guardrails = RAGGuardrailsManager(config=config)

    print(f"[eval] Loading {len(pdf_paths)} PDFs from {os.path.dirname(gold_path)}")
    pages = load_corpus(pdf_paths)
    page_map = {(p.metadata["source"], p.metadata["page"]): p.page_content for p in pages}
    chunks = chunk_documents(pages, chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
    embeddings = build_embeddings(config.embedding_model_id, device=device)
    vectorstore = build_faiss(chunks, embeddings, index_dir_for(chunks, config, config.chunk_size))
    bm25_ret = LocalBM25Retriever(chunks)

    answerable, skipped = select_items(gold["items"], chunks)
    items = (answerable + [item for item in gold["items"] if not is_answerable(item)])[:args.limit]

    print(f"[eval] Retrieving context for {len(items)} questions")
    prepared = [prepare_question(item["question"], vectorstore, bm25_ret, base_registry, config, guardrails, page_map)
                for item in items]
    rows = [{"id": item.get("id", ""), "question": item["question"], "expected_answer": item["expected_answer"],
             "answerable": is_answerable(item), "context_pages": prep["context_pages"]}
            for item, prep in zip(items, prepared)]

    embedder = base_registry.load_embedding_model()
    summaries = {}
    for model in args.models:
        print(f"\n[eval] Model: {model}")
        registry, limiter = registry_for(model, base_registry, config)
        for item, prep, row in zip(items, prepared, rows):
            result = answer_question(item["question"], prep, registry, guardrails, limiter)
            result["scores"] = score_answer(item, result, prep, embedder)
            row[model] = result
            print(f"  {row['id']:9}: {(result.get('error') or result['answer'])[:110]!r}")
        summaries[model] = summarize(rows, model)

    print("\n" + summary_table(summaries))

    in_context = [bool(gold_pages(item) & set(prep["context_pages"])) for item, prep in zip(items, prepared) if is_answerable(item)]
    report = {
        "configuration": {
            "gold_path": gold_path,
            "documents": gold["documents"],
            "models": args.models,
            "embedding_model": config.embedding_model_id,
            "reranker_model": config.reranker_model_id,
            "chunk_size": config.chunk_size,
            "final_k": config.final_k,
            "max_context_chars": config.max_parent_context_chars,
            "answerable": sum(r["answerable"] for r in rows),
            "unanswerable": sum(not r["answerable"] for r in rows),
            "gold_page_in_context": rate(in_context)
        },
        "summary": summaries,
        "skipped": skipped,
        "rows": rows
    }
    json_path, md_path = write_report(report, report_dir)
    print(f"\n[REPORT] {json_path}\n[REPORT] {md_path}")


def rescore(gold: Dict[str, Any], report_dir: str):
    with open(os.path.join(report_dir, "model_comparison.json"), encoding="utf-8") as f:
        report = json.load(f)
    items = {item.get("id", ""): item for item in gold["items"]}
    config = RAGConfig()
    embedder = ModelRegistry(config, device=get_system_device_info(config)["device"]).load_embedding_model()

    models = report["configuration"]["models"]
    for row in report["rows"]:
        prepared = {"context_pages": [tuple(page) for page in row["context_pages"]]}
        for model in models:
            row[model]["scores"] = score_answer(items[row["id"]], row[model], prepared, embedder)
    report["summary"] = {model: summarize(report["rows"], model) for model in models}

    print(summary_table(report["summary"]))
    json_path, md_path = write_report(report, report_dir)
    print(f"\n[REPORT] {json_path}\n[REPORT] {md_path}")


if __name__ == "__main__":
    main()
