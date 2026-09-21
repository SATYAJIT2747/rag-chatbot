import os
import sqlite3
import time
from typing import Dict, Any, List, Optional
from datetime import datetime

def calculate_llm_cost(
    model_name: str,
    input_tokens: int,
    output_tokens: int,
    pricing_map: Optional[Dict[str, Dict[str, float]]] = None
) -> float:
    if pricing_map is None:
        pricing_map = {
            "gemini-flash-lite-latest": {"input_per_1m": 0.075, "output_per_1m": 0.30},
            "gemini-1.5-flash": {"input_per_1m": 0.075, "output_per_1m": 0.30},
            "default": {"input_per_1m": 0.10, "output_per_1m": 0.40}
        }
    
    rates = pricing_map.get(model_name) or pricing_map.get("default", {"input_per_1m": 0.10, "output_per_1m": 0.40})
    input_cost = (input_tokens / 1_000_000.0) * rates["input_per_1m"]
    output_cost = (output_tokens / 1_000_000.0) * rates["output_per_1m"]
    return round(input_cost + output_cost, 6)

class MetricsTracker:
    def __init__(self, db_path: str = "artifacts/rag_metrics.db"):
        self.db_path = db_path
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        except Exception:
            pass
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS request_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp REAL NOT NULL,
                        datetime_str TEXT NOT NULL,
                        document_id TEXT NOT NULL,
                        user_id TEXT NOT NULL,
                        model_name TEXT NOT NULL,
                        input_tokens INTEGER NOT NULL,
                        output_tokens INTEGER NOT NULL,
                        total_tokens INTEGER NOT NULL,
                        latency_ms REAL NOT NULL,
                        retrieval_latency_ms REAL NOT NULL,
                        llm_latency_ms REAL NOT NULL,
                        computed_cost_usd REAL NOT NULL,
                        cache_hit INTEGER NOT NULL,
                        cache_type TEXT NOT NULL,
                        request_category TEXT NOT NULL,
                        num_retrieved_chunks INTEGER NOT NULL,
                        token_count_type TEXT NOT NULL
                    )
                """)
                conn.commit()
        except Exception as e:
            print(f"[MetricsTracker] Warning: DB Init failed - {e}")

    def log_request(
        self,
        document_id: str,
        user_id: str,
        model_name: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        retrieval_latency_ms: float,
        llm_latency_ms: float,
        computed_cost_usd: float,
        cache_hit: bool,
        cache_type: str,
        request_category: str = "pdf_qa",
        num_retrieved_chunks: int = 0,
        token_count_type: str = "actual"
    ):
        now_ts = time.time()
        now_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        total_tokens = input_tokens + output_tokens
        cache_hit_int = 1 if cache_hit else 0

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO request_metrics (
                        timestamp, datetime_str, document_id, user_id, model_name,
                        input_tokens, output_tokens, total_tokens, latency_ms,
                        retrieval_latency_ms, llm_latency_ms, computed_cost_usd,
                        cache_hit, cache_type, request_category, num_retrieved_chunks, token_count_type
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    now_ts, now_dt, document_id, user_id, model_name,
                    input_tokens, output_tokens, total_tokens, latency_ms,
                    retrieval_latency_ms, llm_latency_ms, computed_cost_usd,
                    cache_hit_int, cache_type, request_category, num_retrieved_chunks, token_count_type
                ))
                conn.commit()
        except Exception as e:
            print(f"[MetricsTracker] Warning: Metrics logging failed - {e}")

    def get_summary_stats(self, pricing_map: Optional[Dict[str, Dict[str, float]]] = None) -> Dict[str, Any]:
        default_stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "exact_hits": 0,
            "semantic_hits": 0,
            "cache_misses": 0,
            "cache_hit_rate": 0.0,
            "exact_hit_rate": 0.0,
            "semantic_hit_rate": 0.0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_tokens": 0,
            "total_cost_usd": 0.0,
            "estimated_cost_saved_usd": 0.0,
            "avg_total_latency_ms": 0.0,
            "avg_llm_latency_ms": 0.0,
            "avg_retrieval_latency_ms": 0.0,
            "llm_calls_avoided": 0
        }
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT
                        COUNT(*) as total_requests,
                        SUM(CASE WHEN cache_hit = 1 THEN 1 ELSE 0 END) as cache_hits,
                        SUM(CASE WHEN cache_type = 'exact' THEN 1 ELSE 0 END) as exact_hits,
                        SUM(CASE WHEN cache_type = 'semantic' THEN 1 ELSE 0 END) as semantic_hits,
                        SUM(CASE WHEN cache_hit = 0 THEN 1 ELSE 0 END) as cache_misses,
                        SUM(input_tokens) as total_input_tokens,
                        SUM(output_tokens) as total_output_tokens,
                        SUM(total_tokens) as total_tokens,
                        SUM(computed_cost_usd) as total_cost_usd,
                        AVG(latency_ms) as avg_total_latency_ms,
                        AVG(llm_latency_ms) as avg_llm_latency_ms,
                        AVG(retrieval_latency_ms) as avg_retrieval_latency_ms
                    FROM request_metrics
                """)
                row = cursor.fetchone()
                if not row or row["total_requests"] == 0:
                    return default_stats

                total = row["total_requests"] or 0
                c_hits = row["cache_hits"] or 0
                e_hits = row["exact_hits"] or 0
                s_hits = row["semantic_hits"] or 0
                c_misses = row["cache_misses"] or 0

                # Estimate cost saved: Average cost per live LLM generation * number of avoided calls
                cursor.execute("SELECT AVG(computed_cost_usd) as avg_miss_cost FROM request_metrics WHERE cache_hit = 0")
                miss_row = cursor.fetchone()
                avg_miss_cost = miss_row["avg_miss_cost"] if miss_row and miss_row["avg_miss_cost"] else 0.0
                estimated_savings = round(avg_miss_cost * c_hits, 6)

                return {
                    "total_requests": total,
                    "cache_hits": c_hits,
                    "exact_hits": e_hits,
                    "semantic_hits": s_hits,
                    "cache_misses": c_misses,
                    "cache_hit_rate": round((c_hits / total) * 100.0, 1),
                    "exact_hit_rate": round((e_hits / total) * 100.0, 1),
                    "semantic_hit_rate": round((s_hits / total) * 100.0, 1),
                    "total_input_tokens": row["total_input_tokens"] or 0,
                    "total_output_tokens": row["total_output_tokens"] or 0,
                    "total_tokens": row["total_tokens"] or 0,
                    "total_cost_usd": round(row["total_cost_usd"] or 0.0, 4),
                    "estimated_cost_saved_usd": estimated_savings,
                    "avg_total_latency_ms": round(row["avg_total_latency_ms"] or 0.0, 1),
                    "avg_llm_latency_ms": round(row["avg_llm_latency_ms"] or 0.0, 1),
                    "avg_retrieval_latency_ms": round(row["avg_retrieval_latency_ms"] or 0.0, 1),
                    "llm_calls_avoided": c_hits
                }
        except Exception as e:
            print(f"[MetricsTracker] Warning: Summary stats calculation failed - {e}")
            return default_stats

    def get_recent_requests(self, limit: int = 50) -> List[Dict[str, Any]]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, datetime_str, document_id, model_name, cache_type, cache_hit,
                           input_tokens, output_tokens, total_tokens, latency_ms, computed_cost_usd, request_category
                    FROM request_metrics
                    ORDER BY id DESC
                    LIMIT ?
                """, (limit,))
                rows = cursor.fetchall()
                return [dict(r) for r in rows]
        except Exception as e:
            print(f"[MetricsTracker] Warning: Get recent requests failed - {e}")
            return []
