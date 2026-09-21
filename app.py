import os
import sys
import time
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Add workspace and pdf_rag_bot root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import RAGConfig, get_system_device_info
from src.models import ModelRegistry
from src.document_loader import (
    ensure_sample_pdf,
    load_pdf,
    extract_pages,
    ocr_page,
    build_documents,
    build_page_map
)
from src.indexer import (
    chunk_documents,
    build_embeddings,
    build_faiss,
    build_bm25
)
from src.pipeline import RAGPipeline
from src.evaluator import DEFAULT_EVAL_SUITE, evaluate_retrieval_suite

# Set Streamlit page config
st.set_page_config(
    page_title="Advanced RAG Document Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished, professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
    }
    .citation-badge {
        display: inline-block;
        background-color: #EEF2FF;
        color: #4F46E5;
        border: 1px solid #C7D2FE;
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 4px;
    }
    .source-box {
        background-color: #F8FAFC;
        border-left: 4px solid #4F46E5;
        padding: 10px 14px;
        border-radius: 4px;
        margin-bottom: 8px;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# CACHED RESOURCE LOADERS
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_config() -> RAGConfig:
    cfg = RAGConfig()
    ensure_sample_pdf(cfg.pdf_path)
    return cfg

@st.cache_resource(show_spinner=False)
def get_device() -> dict:
    cfg = load_config()
    return get_system_device_info(cfg)

@st.cache_resource(show_spinner=False)
def get_model_registry() -> ModelRegistry:
    cfg = load_config()
    dev_info = get_device()
    registry = ModelRegistry(cfg, device=dev_info["device"])
    return registry

@st.cache_resource(show_spinner=False)
def process_and_index_document(pdf_path: str, force_rebuild: bool = False):
    """Parses, normalizes, chunks, and indexes a PDF into FAISS & BM25 with OCR Fallback."""
    cfg = load_config()
    dev_info = get_device()
    
    raw_doc = load_pdf(pdf_path)
    extracted = extract_pages(raw_doc, pdf_path)
    processed = [ocr_page(p, raw_doc) for p in extracted]
    raw_doc.close()
    
    docs = build_documents(processed)
    page_map = build_page_map(docs)
    chunks = chunk_documents(docs, chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap)
    
    embeddings = build_embeddings(cfg.embedding_model_id, dev_info["device"])
    vectorstore = build_faiss(chunks, embeddings, cfg.faiss_dir, force_rebuild=force_rebuild)
    bm25 = build_bm25(chunks, cfg.bm25_path, force_rebuild=force_rebuild)
    
    return {
        "num_pages": len(docs),
        "num_chunks": len(chunks),
        "docs": docs,
        "chunks": chunks,
        "page_map": page_map,
        "vectorstore": vectorstore,
        "bm25": bm25
    }

# ---------------------------------------------------------
# SESSION STATE INITIALIZATION
# ---------------------------------------------------------
config = load_config()
device_info = get_device()
registry = get_model_registry()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_pdf_path" not in st.session_state:
    st.session_state.current_pdf_path = config.pdf_path
if "latest_debug_payload" not in st.session_state:
    st.session_state.latest_debug_payload = None
if "latency_history" not in st.session_state:
    st.session_state.latency_history = []
if "eval_results" not in st.session_state:
    st.session_state.eval_results = None

# Ensure base document is processed
if "index_data" not in st.session_state or st.session_state.get("need_reindex", False):
    with st.spinner("Initializing indexes, embeddings, and OCR layer..."):
        st.session_state.index_data = process_and_index_document(st.session_state.current_pdf_path)
        st.session_state.need_reindex = False

index_data = st.session_state.index_data

pipeline = RAGPipeline(
    config=config,
    registry=registry,
    vectorstore=index_data["vectorstore"],
    bm25_retriever=index_data["bm25"],
    page_map=index_data["page_map"]
)

# ---------------------------------------------------------
# SIDEBAR NAVIGATION & CONTROLS
# ---------------------------------------------------------
with st.sidebar:
    st.title("📄 Advanced RAG")
    st.caption(f"Device: **{device_info['device'].upper()}** ({device_info['gpu_name']})")
    
    # Main Navigation
    page = st.radio(
        "Navigation",
        ["💬 Chat", "💰 Cost & Usage", "🔍 Retrieval Debug", "📊 Evaluation", "📈 Analytics", "⚙️ Settings"],
        index=0
    )
    
    st.markdown("---")
    st.subheader("📁 Knowledge Base")
    
    uploaded_file = st.file_uploader("Upload PDF Document", type=["pdf"])
    if uploaded_file is not None:
        save_path = os.path.join(config.artifacts_dir, uploaded_file.name)
        if save_path != st.session_state.current_pdf_path:
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.session_state.current_pdf_path = save_path
            with st.spinner("Processing new document with OCR fallback..."):
                st.session_state.index_data = process_and_index_document(save_path, force_rebuild=True)
                st.session_state.latest_debug_payload = None
                st.session_state.eval_results = None
            st.success("🟢 Document Ready")
            st.rerun()
            
    st.info(f"**Document**: `{os.path.basename(st.session_state.current_pdf_path)}`\n\n"
            f"**Pages**: {index_data['num_pages']} | **Chunks**: {index_data['num_chunks']}")
    
    st.markdown("---")
    st.subheader("⚙️ Retrieval & Cache Settings")
    
    mode_selection = st.radio(
        "Retrieval Mode",
        ["Hybrid", "Hybrid + Query Rewrite", "Dense", "BM25"],
        index=0
    )
    
    use_reranker = st.checkbox("Cross-Encoder Reranker", value=True)
    use_parent_ctx = st.checkbox("Parent Context", value=True)
    
    enable_cache_ui = st.checkbox("Enable Response Cache", value=getattr(config, "enable_cache", True))
    config.enable_cache = enable_cache_ui
    
    with st.expander("🛡️ Guardrail Controls"):
        config.enable_prompt_injection_guard = st.checkbox("Prompt Injection Guard", value=getattr(config, "enable_prompt_injection_guard", True))
        config.enable_scope_guard = st.checkbox("PDF Scope Guard", value=getattr(config, "enable_scope_guard", True))
        config.enable_grounding_guard = st.checkbox("Output Grounding Guard", value=getattr(config, "enable_grounding_guard", True))
        config.max_query_length = st.number_input("Max Query Length", value=getattr(config, "max_query_length", 2000), step=100)

    with st.expander("Advanced K & Cache Settings"):
        retrieval_k = st.slider("Retrieval K (Candidates)", min_value=5, max_value=40, value=20, step=5)
        rerank_k = st.slider("Rerank K (Cross-Encoder)", min_value=5, max_value=20, value=10, step=1)
        final_k = st.slider("Final Context K", min_value=1, max_value=10, value=5, step=1)
        cache_threshold = st.slider("Semantic Cache Threshold", min_value=0.70, max_value=0.99, value=getattr(config, "semantic_cache_threshold", 0.82), step=0.01)
        config.semantic_cache_threshold = cache_threshold
        
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------
# 1. CHAT PAGE
# ---------------------------------------------------------
if page == "💬 Chat":
    st.markdown('<div class="main-header">📄 Advanced RAG Document Assistant</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="sub-header">Grounded Question Answering on <b>{os.path.basename(st.session_state.current_pdf_path)}</b> | Mode: <code>{mode_selection}</code> | Status: 🟢 Ready</div>',
        unsafe_allow_html=True
    )
    
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("cache_type"):
                ctype = msg["cache_type"]
                if ctype == "exact":
                    st.caption("⚡ **Exact Cache Hit** (0ms LLM, $0.000)")
                elif ctype == "semantic":
                    st.caption("🧠 **Semantic Cache Hit** (0ms LLM, $0.000)")
                else:
                    st.caption("🔥 **Live LLM Generation**")
            if msg.get("citations"):
                citation_html = "".join([f'<span class="citation-badge">Page {p}</span>' for p in msg["citations"]])
                st.markdown(f"**Citations:** {citation_html}", unsafe_allow_html=True)
            if msg.get("sources"):
                with st.expander(f"📚 Retrieved Sources ({len(msg['sources'])} chunks)"):
                    for idx, src in enumerate(msg["sources"], 1):
                        st.markdown(f"""
                        <div class="source-box">
                            <b>Source {idx} — Page {src['page']}</b> (ID: <code>{src['chunk_id']}</code>)<br>
                            {src['content']}
                        </div>
                        """, unsafe_allow_html=True)

    # User Input
    if user_query := st.chat_input("Ask a question about the document..."):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)
            
        with st.chat_message("assistant"):
            status_container = st.status("Executing Advanced RAG Pipeline...", expanded=True)
            
            # Map UI options to pipeline parameters
            is_hybrid = "Hybrid" in mode_selection
            is_dense = mode_selection == "Dense"
            ret_mode = "hybrid" if is_hybrid else ("dense" if is_dense else "bm25")
            use_qr = "Query Rewrite" in mode_selection
            
            status_container.write("🔍 Checking Exact & Semantic Cache...")
            time.sleep(0.05)
            
            if use_reranker:
                status_container.write("⚖️ Scoring top candidates with Cross-Encoder (BAAI/bge-reranker-base)...")
                
            status_container.write("🧠 Generating grounded answer with Gemini LLM...")
            
            try:
                response = pipeline.run(
                    query=user_query,
                    retrieval_mode=ret_mode,
                    use_query_rewrite=use_qr,
                    use_reranker=use_reranker,
                    return_parent_context=use_parent_ctx,
                    retrieval_k=retrieval_k,
                    rerank_k=rerank_k,
                    final_k=final_k
                )
                
                cache_status = response.get("cache_type", "miss")
                if cache_status == "exact":
                    status_container.update(label="⚡ Exact Cache Hit!", state="complete", expanded=False)
                elif cache_status == "semantic":
                    status_container.update(label="🧠 Semantic Cache Hit!", state="complete", expanded=False)
                else:
                    status_container.update(label="✅ Answer Generated!", state="complete", expanded=False)
                
                # Display Assistant Answer
                st.markdown(response["answer"])
                
                if cache_status == "exact":
                    st.caption("⚡ **Exact Cache Hit** (0ms LLM, $0.000)")
                elif cache_status == "semantic":
                    st.caption("🧠 **Semantic Cache Hit** (0ms LLM, $0.000)")
                else:
                    st.caption("🔥 **Live LLM Generation**")
                
                if response["citations"]:
                    citation_html = "".join([f'<span class="citation-badge">Page {p}</span>' for p in response["citations"]])
                    st.markdown(f"**Citations:** {citation_html}", unsafe_allow_html=True)
                    
                if response["sources"]:
                    with st.expander(f"📚 Retrieved Sources ({len(response['sources'])} chunks)"):
                        for idx, src in enumerate(response["sources"], 1):
                            st.markdown(f"""
                            <div class="source-box">
                                <b>Source {idx} — Page {src['page']}</b> (ID: <code>{src['chunk_id']}</code>)<br>
                                {src['content']}
                            </div>
                            """, unsafe_allow_html=True)
                            
                # Save to session history & debug cache
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response["answer"],
                    "citations": response["citations"],
                    "sources": response["sources"],
                    "cache_type": cache_status
                })
                st.session_state.latest_debug_payload = {
                    "original_query": user_query,
                    "rewritten_query": response.get("rewritten_query"),
                    **response.get("debug_payload", {})
                }
                st.session_state.latency_history.append({
                    "query": user_query,
                    "timestamp": time.strftime("%H:%M:%S"),
                    **response["latency_breakdown"]
                })
                
            except Exception as e:
                status_container.update(label="❌ Pipeline Error", state="error", expanded=True)
                st.error(f"⚠️ Error executing query: {str(e)}")

# ---------------------------------------------------------
# 2. COST & USAGE PAGE
# ---------------------------------------------------------
elif page == "💰 Cost & Usage":
    st.markdown('<div class="main-header">💰 Cost & Usage Observability</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Production telemetry tracking response caching, token usage, LLM costs, and latency metrics.</div>', unsafe_allow_html=True)
    
    stats = pipeline.metrics_tracker.get_summary_stats(getattr(config, "model_pricing", None))
    
    # ---------------------------------------------------------
    # TOP METRICS DASHBOARD
    # ---------------------------------------------------------
    st.markdown("### 📊 Key Performance Indicators")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Requests", f"{stats['total_requests']}")
    m2.metric("Cache Hit Rate", f"{stats['cache_hit_rate']}%")
    m3.metric("Exact Cache Hits", f"{stats['exact_hits']}")
    m4.metric("Semantic Cache Hits", f"{stats['semantic_hits']}")
    m5.metric("Cache Misses", f"{stats['cache_misses']}")
    
    st.markdown("---")
    m6, m7, m8, m9, m10 = st.columns(5)
    m6.metric("Total Input Tokens", f"{stats['total_input_tokens']:,}")
    m7.metric("Total Output Tokens", f"{stats['total_output_tokens']:,}")
    m8.metric("Total LLM Cost", f"${stats['total_cost_usd']:.4f}")
    m9.metric("Est. Cost Saved", f"${stats['estimated_cost_saved_usd']:.4f}", delta=f"{stats['llm_calls_avoided']} calls avoided")
    m10.metric("Avg Total Latency", f"{stats['avg_total_latency_ms']} ms")
    
    st.markdown("---")
    l1, l2 = st.columns(2)
    l1.metric("Avg LLM Latency", f"{stats['avg_llm_latency_ms']} ms")
    l2.metric("Avg Retrieval Latency", f"{stats['avg_retrieval_latency_ms']} ms")

    # ---------------------------------------------------------
    # GUARDRAILS SECURITY TELEMETRY
    # ---------------------------------------------------------
    st.markdown("---")
    st.markdown("### 🛡️ Guardrails Security & Telemetry")
    g1, g2, g3, g4, g5 = st.columns(5)
    g1.metric("Total Blocked", f"{stats.get('total_blocked', 0)}")
    g2.metric("Length Violations", f"{stats.get('blocked_length', 0)}")
    g3.metric("Injection Blocks", f"{stats.get('blocked_injection', 0)}")
    g4.metric("Out-of-Scope Blocks", f"{stats.get('blocked_scope', 0)}")
    g5.metric("Grounding Failures", f"{stats.get('blocked_ungrounded', 0)}")

    # ---------------------------------------------------------
    # RECENT REQUESTS TABLE
    # ---------------------------------------------------------
    st.markdown("---")
    st.markdown("### 📋 Recent Requests Telemetry")
    recent = pipeline.metrics_tracker.get_recent_requests(limit=50)
    if recent:
        df_recent = pd.DataFrame(recent)
        df_display = pd.DataFrame({
            "Timestamp": df_recent["datetime_str"],
            "Model": df_recent["model_name"],
            "PDF Document": df_recent["document_id"],
            "Cache Status": df_recent["cache_type"].str.upper(),
            "Input Tokens": df_recent["input_tokens"],
            "Output Tokens": df_recent["output_tokens"],
            "Total Tokens": df_recent["total_tokens"],
            "Latency (ms)": df_recent["latency_ms"].round(1),
            "Cost ($)": df_recent["computed_cost_usd"].apply(lambda x: f"${x:.5f}"),
            "Category": df_recent["request_category"]
        })
        st.dataframe(df_display, use_container_width=True, hide_index=True)
    else:
        st.info("No request metrics recorded yet. Ask a question on the **Chat** page to log telemetry!")

    # ---------------------------------------------------------
    # CHARTS & VISUALIZATIONS
    # ---------------------------------------------------------
    st.markdown("---")
    st.markdown("### 📈 Usage & Cost Breakdown")
    c_chart1, c_chart2 = st.columns(2)
    
    with c_chart1:
        st.markdown("#### Cache Hit vs Miss Distribution")
        if stats["total_requests"] > 0:
            df_pie = pd.DataFrame([
                {"Category": "Exact Cache Hit", "Count": stats["exact_hits"]},
                {"Category": "Semantic Cache Hit", "Count": stats["semantic_hits"]},
                {"Category": "Cache Miss (LLM Call)", "Count": stats["cache_misses"]}
            ])
            fig_pie = px.pie(
                df_pie,
                values="Count",
                names="Category",
                color="Category",
                hole=0.4,
                color_discrete_map={
                    "Exact Cache Hit": "#10B981",
                    "Semantic Cache Hit": "#06B6D4",
                    "Cache Miss (LLM Call)": "#F59E0B"
                }
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.caption("No requests available for distribution chart.")

    with c_chart2:
        st.markdown("#### Cumulative LLM Cost ($)")
        if recent:
            df_trend = pd.DataFrame(recent).iloc[::-1].copy()
            df_trend["Cumulative Cost ($)"] = df_trend["computed_cost_usd"].cumsum()
            fig_cost = px.line(
                df_trend,
                x="datetime_str",
                y="Cumulative Cost ($)",
                markers=True,
                title="LLM Cost Accumulation Over Time",
                color_discrete_sequence=["#6366F1"]
            )
            fig_cost.update_layout(xaxis_title="Time", yaxis_title="Cost ($)")
            st.plotly_chart(fig_cost, use_container_width=True)
        else:
            st.caption("No cost trend available yet.")

# ---------------------------------------------------------
# 2. RETRIEVAL DEBUG PAGE
# ---------------------------------------------------------
elif page == "🔍 Retrieval Debug":
    st.markdown('<div class="main-header">🔍 Retrieval Debug Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Full visibility into Query Transformation, BM25, Dense Search, RRF Fusion, and Cross-Encoder Reranking.</div>', unsafe_allow_html=True)
    
    payload = st.session_state.latest_debug_payload
    if not payload:
        st.info("ℹ️ No query executed yet. Ask a question on the **Chat** page to inspect the retrieval pipeline.")
    else:
        st.markdown(f"### 🎯 Original Query: `{payload.get('original_query', '')}`")
        if payload.get("rewritten_query"):
            st.success(f"**Rewritten Query:** `{payload['rewritten_query']}`")
        else:
            st.caption("Query rewriting was disabled for this run.")
            
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 1. BM25 Lexical Results (Top 10)")
            bm25_raw = payload.get("bm25_raw", [])
            if bm25_raw:
                df_bm25 = pd.DataFrame([
                    {"Rank": rank, "Page": doc.metadata.get("page"), "Score": round(score, 3), "Snippet": doc.page_content[:80].replace('\n', ' ') + "..."}
                    for doc, score, rank in bm25_raw[:10]
                ])
                st.dataframe(df_bm25, use_container_width=True, hide_index=True)
            else:
                st.write("No BM25 data.")
                
        with col2:
            st.markdown("#### 2. Dense Vector Results (Top 10)")
            dense_raw = payload.get("dense_raw", [])
            if dense_raw:
                df_dense = pd.DataFrame([
                    {"Rank": rank, "Page": doc.metadata.get("page"), "Score": round(score, 3), "Snippet": doc.page_content[:80].replace('\n', ' ') + "..."}
                    for doc, score, rank in dense_raw[:10]
                ])
                st.dataframe(df_dense, use_container_width=True, hide_index=True)
            else:
                st.write("No Dense vector data.")
                
        st.markdown("#### 3. Reciprocal Rank Fusion (RRF k=60)")
        rrf_debug = payload.get("rrf_debug", [])
        if rrf_debug:
            df_rrf = pd.DataFrame(rrf_debug)
            st.dataframe(df_rrf, use_container_width=True, hide_index=True)
            
        st.markdown("#### 4. Cross-Encoder Reranking (Before vs After)")
        rerank_debug = payload.get("rerank_debug", [])
        if rerank_debug:
            df_rerank = pd.DataFrame(rerank_debug)
            st.dataframe(df_rerank, use_container_width=True, hide_index=True)
        else:
            st.caption("Reranker was not invoked or no candidates.")
            
        st.markdown("#### 5. Final Context Sent to LLM")
        ctx = payload.get("context_used", "")
        if ctx:
            st.text_area("Prompt Context", value=ctx, height=250, disabled=True)

# ---------------------------------------------------------
# 3. EVALUATION PAGE
# ---------------------------------------------------------
elif page == "📊 Evaluation":
    st.markdown('<div class="main-header">📊 Retrieval Evaluation Benchmark</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Live benchmark comparing Sparse BM25, Dense Vector, Hybrid RRF, and Cross-Encoder Reranker.</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"**Target Document**: `{os.path.basename(st.session_state.current_pdf_path)}` ({index_data['num_pages']} pages)")
    with col2:
        run_eval_btn = st.button("🚀 Run Live Evaluation", use_container_width=True, type="primary")
        
    if run_eval_btn:
        with st.spinner("Running comprehensive evaluation over test queries..."):
            eval_res = evaluate_retrieval_suite(
                DEFAULT_EVAL_SUITE,
                index_data["vectorstore"],
                index_data["bm25"],
                registry
            )
            st.session_state.eval_results = eval_res
            st.success("Evaluation complete!")
            
    if st.session_state.eval_results:
        res = st.session_state.eval_results
        st.markdown(f"### Benchmark Results ({res['total_queries']} Ground-Truth Queries)")
        
        eval_rows = []
        for method, metrics in res["methods"].items():
            eval_rows.append({"Retrieval Strategy": method, **metrics})
        df_eval = pd.DataFrame(eval_rows)
        st.dataframe(df_eval, use_container_width=True, hide_index=True)
        
        fig = px.bar(
            df_eval,
            x="Retrieval Strategy",
            y=["Recall@5", "Recall@10", "MRR@10", "nDCG@10"],
            barmode="group",
            title="Retrieval Metrics Comparison",
            color_discrete_sequence=["#4F46E5", "#06B6D4", "#10B981", "#F59E0B"]
        )
        fig.update_layout(yaxis_range=[0, 1.05], legend_title_text="Metric")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No evaluation results available. Click **'Run Live Evaluation'** above to generate benchmark metrics.")

# ---------------------------------------------------------
# 4. ANALYTICS PAGE
# ---------------------------------------------------------
elif page == "📈 Analytics":
    st.markdown('<div class="main-header">📈 System Performance & Latency Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Latency profiling across all individual pipeline stages.</div>', unsafe_allow_html=True)
    
    history = st.session_state.latency_history
    if history:
        total_reqs = len(history)
        avg_total = sum(h.get("Total Latency (sec)", 0) for h in history) / total_reqs
        avg_llm = sum(h.get("LLM Generation (sec)", 0) for h in history) / total_reqs
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Requests", total_reqs)
        c2.metric("Avg Total Latency", f"{avg_total:.2f}s")
        c3.metric("Avg LLM Gen", f"{avg_llm:.2f}s")
        
        latest = history[-1]
        st.markdown("### ⏱️ Latest Query Latency Breakdown")
        
        latency_stages = []
        for k, v in latest.items():
            if "ms" in k:
                latency_stages.append((k.replace(" (ms)", ""), v / 1000.0))
            elif "sec" in k and "Total" not in k:
                latency_stages.append((k.replace(" (sec)", ""), v))
                
        df_latency = pd.DataFrame(latency_stages, columns=["Stage", "Duration (sec)"])
        fig_lat = px.bar(
            df_latency,
            x="Duration (sec)",
            y="Stage",
            orientation="h",
            text="Duration (sec)",
            color="Stage",
            color_discrete_sequence=px.colors.qualitative.Prism
        )
        fig_lat.update_traces(texttemplate='%{text:.3f}s', textposition='outside')
        st.plotly_chart(fig_lat, use_container_width=True)
    else:
        st.info("No query latency data recorded yet. Run a query in Chat to view live telemetry.")
        
    st.markdown("---")
    st.markdown("### 🔍 LangSmith Observability")
    tracing_active = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    project_name = os.getenv("LANGCHAIN_PROJECT", "advanced-pdf-rag")
    
    col_ls1, col_ls2 = st.columns([3, 1])
    with col_ls1:
        st.write(f"**Tracing Status**: {'🟢 Active' if tracing_active else '⚪ Inactive (Local fallback)'}")
        st.write(f"**Project**: `{project_name}`")
    with col_ls2:
        if tracing_active:
            st.link_button("🔎 Open LangSmith Trace", "https://smith.langchain.com", use_container_width=True)
        else:
            st.button("🔎 Open LangSmith Trace", disabled=True, use_container_width=True)

# ---------------------------------------------------------
# 5. SETTINGS PAGE
# ---------------------------------------------------------
elif page == "⚙️ Settings":
    st.markdown('<div class="main-header">⚙️ System Configuration</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Manage model configurations, retrieval hyperparameters, and runtime preferences.</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🤖 Model Identifiers")
        st.text_input("Embedding Model", value=config.embedding_model_id, disabled=True)
        st.text_input("Reranker Model", value=config.reranker_model_id, disabled=True)
        st.text_input("Generation LLM", value=config.llm_model_id, disabled=True)
        
    with col2:
        st.markdown("#### 🛠️ Runtime & Precision")
        st.text_input("Execution Device", value=f"{device_info['device'].upper()} ({device_info['gpu_name']})", disabled=True)
        st.text_input("Available VRAM", value=f"{device_info['vram_gb']} GB", disabled=True)
        st.text_input("System RAM", value=f"{device_info['ram_gb']} GB", disabled=True)
        
    st.markdown("---")
    st.markdown("#### 🔒 Security & Secrets")
    st.write("API keys and secrets are loaded securely from `.env` via `python-dotenv` and never logged or exposed in the UI.")
    if os.getenv("GOOGLE_API_KEY"):
        st.success("🟢 Google GenAI API Key is configured and active.")
    else:
        st.warning("⚠️ GOOGLE_API_KEY not found in environment.")
