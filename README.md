# Production-Grade Advanced PDF RAG Assistant (`pdf_rag_bot`)

A complete, production-ready Advanced Retrieval-Augmented Generation (RAG) system with dual-tier response caching, cost observability, and LangSmith telemetry. Capable of ingesting arbitrary PDF documents (including scanned pages, slide decks, diagrams, tables, and academic lecture notes) with strict cited grounding, sub-second responses, and zero-cost cached generation.

---

## 🏛️ System Architecture

```
                                USER QUERY
                                    │
                         [Query Normalization &
                          PDF SHA256 Document Hash]
                                    │
                                    ▼
                         ┌────────────────────┐
                         │    EXACT CACHE     │ ── Hit ──► Return Cached Answer (0ms LLM, $0)
                         └──────────┬─────────┘
                               Miss │
                                    ▼
                         ┌────────────────────┐
                         │   SEMANTIC CACHE   │ ── Hit ──► Return Cached Answer (0ms LLM, $0)
                         │  (Cosine >= 0.82)  │
                         └──────────┬─────────┘
                               Miss │
                                    ▼
                            HYBRID RETRIEVER
                     ┌──────────────┴──────────────┐
                     ▼                             ▼
              Dense FAISS Index             Sparse BM25 Index
           (BAAI/bge-small-en-v1.5)           (BM25Okapi)
                     │                             │
                     └──────────────┬──────────────┘
                                    │
                     [Reciprocal Rank Fusion (RRF k=60)]
                                    │
                                    ▼ (Top 20 Candidates)
                         CROSS-ENCODER RERANKER
                         (BAAI/bge-reranker-base)
                                    │
                                    ▼ (Top 5 Reranked)
                         PARENT CONTEXT EXPANSION
                                    │
                                    ▼
                          GENERATION LLM & PROMPT
                 (ChatGoogleGenerativeAI: gemini-flash-lite)
                                    │
                                    ▼
                         ┌────────────────────┐
                         │  COST & METRICS    │
                         │    TELEMETRY       │ ──► Persist to SQLite & LangSmith
                         └──────────┬─────────┘
                                    │
                                    ▼
                              CITED ANSWER
                        (Strict [Page X] Grounding)
```

---

## 📁 Project Structure

```
pdf_rag_bot/
│
├── README.md                      # Complete system documentation
├── requirements.txt               # Pinned dependencies
├── .env.example                   # API & LangSmith configuration template
├── main.py                        # Single-command CLI runner & evaluation benchmark
├── app.py                         # 6-Page interactive Streamlit Web UI
│
├── data/
│   └── sample_document.pdf        # Knowledge base PDF
│
├── src/
│   ├── __init__.py                # Module exports
│   ├── config.py                  # System configuration, pricing, & hardware detection
│   ├── document_loader.py         # PyMuPDF + Ligature Decoding + Gemini Vision OCR Fallback
│   ├── indexer.py                 # Chunker, FAISS dense vector store & BM25 sparse indexer
│   ├── models.py                  # Model Registry (BGE Embedder, Cross-Encoder, Gemini LLM)
│   ├── retrieval.py               # Dense, BM25, RRF Hybrid Fusion (k=60), Reranker & Query Rewriter
│   ├── generation.py              # Parent context expansion, prompt construction & token usage extraction
│   ├── caching.py                 # Exact & Semantic Cache Manager (SQLite + SHA256 document hashing)
│   ├── metrics.py                 # Persistent request metrics telemetry & LLM cost calculation engine
│   ├── evaluator.py               # Benchmark evaluation suite (Recall@5, Recall@10, MRR@10, nDCG@10)
│   └── pipeline.py                # High-level RAGPipeline wrapped with caching & LangSmith `@traceable`
│
├── tests/
│   └── test_caching_and_metrics.py# Automated unit & integration tests
│
└── artifacts/                     # Persisted SQLite databases, vector stores & reports
    ├── rag_cache.db               # SQLite exact & semantic response cache database
    ├── rag_metrics.db             # SQLite request telemetry & usage database
    ├── faiss/                     # Vector index cache
    └── bm25/                      # BM25 pickle cache
```

---

## 🚀 Quick Start

### 1. Configure Environment Variables
Create or update your `.env` file:
```bash
GOOGLE_API_KEY=your_gemini_api_key_here

# LangSmith Observability (Optional)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT="Chat-bot"
```

### 2. Run Automated Test Suite
Verify caching, similarity thresholds, document isolation, and cost calculations:
```bash
python -m unittest discover -s tests
```

### 3. Launch Interactive Streamlit UI
```bash
streamlit run app.py
```
*(Or from the main repository directory: `..\.venv\Scripts\streamlit run app.py`)*

---

## 💻 Streamlit UI Navigation

1. **💬 Chat**: Interactive QA with direct `[Page X]` citation badges, source preview expanders, and instant cache badges (`⚡ Exact Cache Hit`, `🧠 Semantic Cache Hit`, `🔥 Live LLM`).
2. **💰 Cost & Usage**: 10 KPI metric cards (Requests, Hit Rate %, Exact/Semantic Hits, Tokens, LLM Cost $, Est. Savings $, Latencies), recent request telemetry table, and Plotly distribution & cumulative cost charts.
3. **🔍 Retrieval Debug**: Side-by-side inspection of BM25 scores, Dense cosine distances, RRF fused rankings, and Cross-Encoder re-scores.
4. **📊 Evaluation**: Live benchmark metrics comparison (Recall@5, Recall@10, MRR@10, nDCG@10).
5. **📈 Analytics**: Real-time latency profiling broken down per pipeline stage (ms/sec).
6. **⚙️ Settings**: Hardware diagnostics, model inspectability, cache threshold slider (`0.82`), and secret configurations.

---

## ⚡ Response Caching & Cost Observability

- **Exact Response Cache**: Computes SHA256 key of `(doc_hash + normalized_query + model_name + prompt_version)`. Identical queries return answers instantly with **0ms LLM latency and $0 cost**.
- **Semantic Response Cache**: Performs cosine similarity search over cached query vectors scoped strictly to the current PDF document hash. Matches above threshold (`0.82`) short-circuit the LLM call with **$0 cost**.
- **Document Version Scoping**: SHA256 PDF content hashing guarantees old cache entries are automatically invalidated when a document is updated or replaced.
- **Cost Calculation**: Automated token usage tracking (`input_tokens`, `output_tokens`) using real provider API metadata or fallback tokenizers, multiplied by per-model pricing configuration.
- **LangSmith Tracing**: Full OpenTelemetry request tracing with dynamic tags (`pdf-rag`, `cache-hit`, `cache-miss`) and metadata attributes (`llm_skipped`, `document_id`, `cache_type`).
