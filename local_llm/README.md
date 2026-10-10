# Local LLM Pipeline (Qwen3.5-4B via llama.cpp)

A second generation path for `pdf_rag_bot` that swaps the Gemini API for a local
Qwen3.5-4B GGUF served by `llama-server`. Everything else is **identical** to the
Gemini pipeline: same `src/` code, the same FAISS + BM25 indexes, RRF fusion,
cross-encoder reranker, parent-context expansion, system prompt, and input/scope/grounding
guardrails. Only the LLM call changes, so the comparison isolates the generator.

```
src/RAGPipeline ──► retrieval ─► rerank ─► context ─► LLM ─► grounding guard
                                                       │
                              Gemini pipeline:  ChatGoogleGenerativeAI (gemini-flash-lite-latest)
                              Local pipeline :  LlamaServerChat ─► http://127.0.0.1:8080/v1 (Qwen3.5-4B-Q8_0)
```

## Files

| File | Purpose |
|---|---|
| `llm_client.py` | `LlamaServerChat`: small OpenAI-compatible client with the same `.invoke()` shape as LangChain chat models. Disables Qwen "thinking" and strips any `<think>` blocks. |
| `pipeline.py` | `build_local_pipeline()` / `local_pipeline_from()`: build a `RAGPipeline` with the local LLM injected into `ModelRegistry`. Separate cache and metrics DBs under `artifacts/local_llm/`. |
| `main.py` | CLI to ask one question with the local LLM. |
| `compare.py` | Runs `DEFAULT_EVAL_SUITE` through Gemini and the local LLM with caching off and writes `artifacts/local_llm/report/gemini_vs_local.{md,json}`. |

## Run

1. Start the model server (from `C:\Users\satya\Desktop\LLM Serve`):
   ```powershell
   .\serve.ps1 -Model 4b-q8
   ```
2. From `pdf_rag_bot/`:
   ```powershell
   # Streamlit UI on the local model (omit RAG_LLM_BACKEND, or set it to "gemini", for Gemini)
   $env:RAG_LLM_BACKEND="local"; ..\.venv\Scripts\streamlit run app.py

   # One question from the CLI
   ..\.venv\Scripts\python.exe -m local_llm.main --pdf artifacts/Lecture1.pdf --query "What does CORE banking stand for?"

   # Gemini vs local, side by side
   ..\.venv\Scripts\python.exe -m local_llm.compare --pdf artifacts/Lecture1.pdf --questions local_llm/questions_lecture1.json
   ```
   Reports are written to `artifacts/local_llm/report/gemini_vs_local_<pdf>_<gguf>.{md,json}`.

> **Use a text-based PDF.** `data/sample_document.pdf` is a scanned, image-only PDF, and Tesseract
> is not installed, so every page extracts as `[Page N: Scanned or image content.]` and **both**
> models (correctly) refuse every question. The comparison therefore uses `Lecture1.pdf`.
> `compare.py` also keeps its own per-PDF index under `artifacts/local_llm/index/`, because
> `src/indexer.py` reuses whatever FAISS/BM25 index is on disk without checking which PDF it was built from.

Settings (environment variables or `.env`):

| Variable | Default | Meaning |
|---|---|---|
| `LOCAL_LLM_BASE_URL` | `http://127.0.0.1:8080/v1` | llama-server endpoint |
| `LOCAL_LLM_MODEL` | `qwen3.5-4b-q8` | Label used for metrics, cache keys, and pricing ($0) |
| `LOCAL_LLM_THINKING` | `false` | Turn on Qwen3.5 reasoning mode (slower, more output tokens) |

## Observations: Gemini vs local Qwen3.5-4B

Setup: `Lecture1.pdf` (Distributed Systems, 24 pages), 9 questions (8 answerable + 1 out-of-scope), hybrid
retrieval + reranker, caching disabled, Qwen thinking off, temperature 0.1, laptop RTX 4050 (6 GB).

| Model | Answered | Avg LLM time | Avg total | Avg out tokens | Throughput | Grounded | Cited expected page | Cost (9 q) |
|---|---|---|---|---|---|---|---|---|
| gemini-flash-lite-latest | 9/9 | **1.19 s** | **1.28 s** | 63 | 58 tok/s | 9/9 | 8/8 | $0.00067 |
| Qwen3.5-4B **Q8_0** (llama.cpp) | 9/9 | 2.67 s | 2.71 s | 75 | 26 tok/s | 9/9 | 8/8 | $0 |

What we saw:

- **Answer quality is on par.** Both models answered all 8 in-scope questions correctly, passed the
  grounding guard 9/9, and cited the expected page every time. Both refused the out-of-scope question
  ("capital of Australia") with the exact refusal string.
- **Citations: Qwen was slightly more precise.** For "daily-use distributed systems", Gemini cited
  `[Pages 3, 4]` while only page 4 holds the list; Qwen cited `[Page 4]`.
- **Qwen is wordier.** It averaged about 18% more output tokens. It adds "Based on the retrieved context",
  bold headings, and extra sentences that weren't asked for. On the computation-model question it gave 92 tokens to Gemini's 40.
  The extra text was still grounded.
- **Gemini is about 2.2x faster end to end** (1.2 s vs 2.7 s per answer). Local throughput was about 26 output tok/s
  on the 4050. That figure includes prompt processing of the parent context, up to about 1k tokens. Retrieval and reranking are the
  same code on both and took about 0.1 s.
- **Cost and privacy favour local.** $0 per query, and no document text leaves the machine. Gemini cost
  about $0.000074 per question. That is tiny, but it needs a network connection, an API key, and rate-limit headroom.
- **Thinking mode must stay off.** Qwen3.5 is a hybrid reasoning model. Its reasoning tokens would multiply latency
  for no gain on extractive RAG answers. The client sends `enable_thinking: false` and strips any `<think>` blocks.
- **Scanned PDFs break both pipelines equally.** On the image-only `sample_document.pdf`, both models refused all
  6 questions, because retrieval had no text to give them. The bottleneck there is OCR, not the LLM.

**Takeaway:** for this workload, a 4B local model at Q8 matches Gemini Flash-Lite on correctness, grounding
and citations, for about 1.5 s more latency per answer and zero API cost. Gemini stays the better default
for speed; the local path suits offline use, private documents, or getting around API quotas.
