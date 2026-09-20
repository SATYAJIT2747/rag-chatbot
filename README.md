# Production-Grade Advanced PDF RAG Assistant (`pdf_rag_bot`)

A complete, production-ready Advanced Retrieval-Augmented Generation (RAG) system capable of ingesting arbitrary PDF documents (including scanned pages, slide decks, diagrams, tables, and academic lecture notes) with strict cited grounding and sub-second responses.

---

## 🏛️ Architecture Overview

```
                      USER QUERY
                          │
                 [Query Reformulation]
                          │
                          ▼
                  HYBRID RETRIEVER
            ┌─────────────┴─────────────┐
            ▼                           ▼
     Dense FAISS Index           Sparse BM25 Index
  (BAAI/bge-small-en-v1.5)         (BM25Okapi)
            │                           │
            └─────────────┬─────────────┘
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
                 GENERATION LLM
           (ChatGoogleGenerativeAI: gemini-flash-lite-latest)
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
├── .env.example                   # API configuration template
├── main.py                        # Single-command CLI test & evaluation runner
├── app.py                         # 5-Page interactive Streamlit Web UI
│
├── data/
│   └── sample_document.pdf        # Target knowledge base PDF
│
├── src/
│   ├── __init__.py                # Module exports
│   ├── config.py                  # System config & hardware device detection
│   ├── document_loader.py         # PyMuPDF + Ligature Decoding + Gemini Vision OCR Fallback
│   ├── indexer.py                 # Recursive chunker, FAISS dense store & BM25 sparse indexer
│   ├── models.py                  # Model Registry (BGE, Cross-Encoder, Gemini LLM)
│   ├── retrieval.py               # Dense, BM25, RRF Hybrid Fusion (k=60), Cross-Encoder & Query Rewriter
│   ├── generation.py              # Parent context expansion, prompt construction & cited answering
│   ├── evaluator.py               # Benchmark evaluation suite (Recall@5, Recall@10, MRR@10, nDCG@10)
│   └── pipeline.py                # High-level unified RAGPipeline interface
│
└── artifacts/                     # Local serialized vector store & demo reports
    ├── faiss/
    ├── bm25/
    └── report/
```

---

## 🚀 Quick Start

### 1. Installation
Ensure dependencies are installed in your Python environment:
```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Create a `.env` file (or copy `.env.example`):
```bash
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 3. Run via CLI (One Command)
Run an end-to-end demonstration, query answering, and retrieval evaluation benchmark:
```bash
python main.py
```

To ask a custom question:
```bash
python main.py --query "What is Reciprocal Rank Fusion (RRF) and the smoothing constant formula?"
```

To test with a different PDF:
```bash
python main.py --pdf "path/to/document.pdf" --query "Summarize the key points"
```

### 4. Launch Interactive Streamlit UI
Launch the full web assistant:
```bash
streamlit run app.py
```
**Features in the UI:**
1. **💬 Chat**: Interactive chat assistant with direct `[Page X]` citation badges and source preview cards.
2. **🔍 Retrieval Debug**: Inspect BM25 scores, Dense cosine distances, RRF fused rankings, and Cross-Encoder re-scores side-by-side.
3. **📊 Evaluation**: Live benchmark metrics comparison (Recall@5, Recall@10, MRR@10, nDCG@10).
4. **📈 Analytics**: Real-time latency profiling broken down per pipeline stage (ms/sec).
5. **⚙️ Settings**: Hardware diagnostics, model inspectability, and secure key configuration.

---

## 💡 Key Highlights
- **Scanned / Image PDF Ingestion**: Automatic detection of scanned or sparse pages with seamless Gemini Vision OCR fallback.
- **Accurate Citations**: Page numbers are preserved across all chunking and OCR layers for guaranteed attribution.
- **Anti-Hallucination Policy**: Returns deterministic refusal when the question cannot be answered from the document context.
