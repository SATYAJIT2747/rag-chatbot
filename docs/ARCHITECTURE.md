# System Architecture Specification (`pdf_rag_bot`)

This document provides a detailed breakdown of the system architecture, flow, components, and security guardrails for the **Advanced PDF RAG Assistant**.

---

## 🏛️ System Architecture Overview

![Advanced PDF RAG System Architecture](architecture.png)

---

## 🧩 Architectural Subsystems

### 1. 📄 100% Local PDF Ingestion Pipeline
- **Upload & Parsing:** PDF documents (including slides, papers, tables, and scanned documents) are ingested locally using **PyMuPDF (`fitz`)**.
- **Ligature Decoding:** Proprietary font PUA mappings and Unicode ligatures (`fi`, `fl`, `ffi`, `ffl`) are decoded into clean standard text strings.
- **Local OCR Engine:** If a page contains sparse or scanned content, **PyTesseract** performs local optical character recognition. **Zero external LLM API calls** are made during document scanning and indexing.
- **Chunking Strategy:** Parent-Child chunking partitions raw page text into smaller child chunks (for vector search precision) while preserving parent page context.

---

### 2. ⚡ Dual-Tier Response Caching Layer
- **Exact Hash Match (`0ms LLM, $0`):** Query strings are normalized and SHA256 hashed. Matches return pre-computed cited answers instantly.
- **Semantic Vector Cache (`0ms LLM, $0`):** Dense query embedding vectors are queried against an isolated SQLite cache index. Queries with cosine similarity $\ge 0.82$ short-circuit the LLM call.
- **SHA256 Document Versioning:** Cache entries are strictly tied to the PDF content hash, ensuring cache safety across document updates.

---

### 3. 🔍 Hybrid Retrieval & Reranking Engine
- **Dense Vector Search:** Uses `BAAI/bge-small-en-v1.5` over a local **FAISS** index for conceptual similarity.
- **Sparse Keyword Search:** Uses **BM25Okapi** with exact token matching for technical terms, part numbers, and abbreviations.
- **Reciprocal Rank Fusion (RRF):** Fuses dense and sparse candidate rankings without score normalization ($k=60$).
- **Cross-Encoder Reranker:** Evaluates top-20 fused candidates through `BAAI/bge-reranker-base` full cross-attention to isolate the top-5 most relevant context chunks.

---

### 4. 🛡️ Modular Security & Quality Guardrails
- **Input Guardrails:** Enforces maximum query length (`MAX_QUERY_LENGTH = 2000`) and blocks empty / whitespace-only queries. Deterministic regex pattern matching intercepts prompt injection and system override attempts.
- **PDF Scope Guard:** Reuses Cross-Encoder candidate scores ($threshold = -4.0$) to block out-of-scope queries before LLM prompting without additional latency.
- **Parent Context Expansion:** Injects full parent page text for top reranked chunks into the LLM prompt.
- **Output Grounding Guard:** Performs n-gram and numeric overlap checks between generated answers and retrieved PDF context to prevent hallucination.

---

### 5. 📊 Cost Observability & Telemetry Logging
- **SQLite Database:** Logs request metrics, latencies, exact/semantic cache hits, guardrail events, token counts, and cost breakdown (`artifacts/rag_metrics.db`).
- **LangSmith Integration:** Full OpenTelemetry tracing with dynamic tags (`pdf-rag`, `cache-hit`, `cache-miss`) and metadata attributes.
