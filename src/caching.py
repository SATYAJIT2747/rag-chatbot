import os
import re
import json
import sqlite3
import hashlib
import time
import numpy as np
from typing import Dict, Any, Optional, Tuple, List

def compute_doc_hash(pdf_path: str) -> str:
    """Computes a SHA-256 hash of the PDF file to ensure cache invalidation on file changes."""
    if not os.path.exists(pdf_path):
        return "unknown_doc_hash"
    try:
        hasher = hashlib.sha256()
        with open(pdf_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()[:16]
    except Exception:
        return "fallback_doc_hash"

def normalize_query(query: str) -> str:
    """Normalizes query string by lowercasing, stripping extra whitespace, and trailing punctuation."""
    q = query.strip().lower()
    q = re.sub(r'\s+', ' ', q)
    q = re.sub(r'[^\w\s]', '', q)
    return q

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    v1 = np.array(vec1, dtype=np.float32)
    v2 = np.array(vec2, dtype=np.float32)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))

class RAGCacheManager:
    def __init__(self, db_path: str = "artifacts/rag_cache.db"):
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
                # 1. Exact Cache Table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS exact_cache (
                        exact_key TEXT PRIMARY KEY,
                        doc_hash TEXT NOT NULL,
                        document_id TEXT NOT NULL,
                        normalized_query TEXT NOT NULL,
                        model_name TEXT NOT NULL,
                        prompt_version TEXT NOT NULL,
                        answer TEXT NOT NULL,
                        citations_json TEXT NOT NULL,
                        sources_json TEXT NOT NULL,
                        created_at REAL NOT NULL
                    )
                """)
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_exact_doc ON exact_cache(doc_hash)")

                # 2. Semantic Cache Table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS semantic_cache (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        doc_hash TEXT NOT NULL,
                        document_id TEXT NOT NULL,
                        original_query TEXT NOT NULL,
                        normalized_query TEXT NOT NULL,
                        query_embedding_json TEXT NOT NULL,
                        model_name TEXT NOT NULL,
                        prompt_version TEXT NOT NULL,
                        answer TEXT NOT NULL,
                        citations_json TEXT NOT NULL,
                        sources_json TEXT NOT NULL,
                        created_at REAL NOT NULL
                    )
                """)
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_semantic_doc_model ON semantic_cache(doc_hash, model_name, prompt_version)")
                conn.commit()
        except Exception as e:
            print(f"[RAGCacheManager] Warning: DB Init failed - {e}")

    def compute_exact_key(
        self,
        doc_hash: str,
        normalized_query: str,
        model_name: str,
        prompt_version: str
    ) -> str:
        raw_key = f"{doc_hash}:{normalized_query}:{model_name}:{prompt_version}"
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    def get_exact(
        self,
        doc_hash: str,
        query: str,
        model_name: str,
        prompt_version: str
    ) -> Optional[Dict[str, Any]]:
        norm_q = normalize_query(query)
        exact_key = self.compute_exact_key(doc_hash, norm_q, model_name, prompt_version)
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT answer, citations_json, sources_json FROM exact_cache WHERE exact_key = ?",
                    (exact_key,)
                )
                row = cursor.fetchone()
                if row:
                    return {
                        "cache_type": "exact",
                        "answer": row["answer"],
                        "citations": json.loads(row["citations_json"]),
                        "sources": json.loads(row["sources_json"])
                    }
        except Exception as e:
            print(f"[RAGCacheManager] Warning: Exact cache lookup error - {e}")
        return None

    def get_semantic(
        self,
        doc_hash: str,
        query: str,
        query_embedding: List[float],
        model_name: str,
        prompt_version: str,
        threshold: float = 0.90
    ) -> Optional[Dict[str, Any]]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, normalized_query, query_embedding_json, answer, citations_json, sources_json
                    FROM semantic_cache
                    WHERE doc_hash = ? AND model_name = ? AND prompt_version = ?
                """, (doc_hash, model_name, prompt_version))
                
                rows = cursor.fetchall()
                if not rows:
                    return None

                best_sim = -1.0
                best_match = None

                for row in rows:
                    cached_vec = json.loads(row["query_embedding_json"])
                    sim = cosine_similarity(query_embedding, cached_vec)
                    if sim > best_sim:
                        best_sim = sim
                        best_match = row

                if best_match and best_sim >= threshold:
                    return {
                        "cache_type": "semantic",
                        "similarity": round(best_sim, 4),
                        "answer": best_match["answer"],
                        "citations": json.loads(best_match["citations_json"]),
                        "sources": json.loads(best_match["sources_json"])
                    }
        except Exception as e:
            print(f"[RAGCacheManager] Warning: Semantic cache lookup error - {e}")
        return None

    def put(
        self,
        doc_hash: str,
        document_id: str,
        query: str,
        query_embedding: Optional[List[float]],
        model_name: str,
        prompt_version: str,
        answer: str,
        citations: List[int],
        sources: List[Dict[str, Any]]
    ):
        if not answer or "does not contain enough information" in answer.lower():
            # Still valid to cache, but omit if empty or total error
            if not answer:
                return

        norm_q = normalize_query(query)
        exact_key = self.compute_exact_key(doc_hash, norm_q, model_name, prompt_version)
        cit_json = json.dumps(citations)
        src_json = json.dumps(sources)
        now = time.time()

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                # Write to exact cache
                cursor.execute("""
                    INSERT OR REPLACE INTO exact_cache
                    (exact_key, doc_hash, document_id, normalized_query, model_name, prompt_version, answer, citations_json, sources_json, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (exact_key, doc_hash, document_id, norm_q, model_name, prompt_version, answer, cit_json, src_json, now))

                # Write to semantic cache if embedding provided
                if query_embedding is not None:
                    emb_json = json.dumps(query_embedding)
                    cursor.execute("""
                        INSERT INTO semantic_cache
                        (doc_hash, document_id, original_query, normalized_query, query_embedding_json, model_name, prompt_version, answer, citations_json, sources_json, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (doc_hash, document_id, query, norm_q, emb_json, model_name, prompt_version, answer, cit_json, src_json, now))

                conn.commit()
        except Exception as e:
            print(f"[RAGCacheManager] Warning: Cache put error - {e}")
