import os
import sys
import unittest
import tempfile
import shutil

# Ensure pdf_rag_bot directory is in Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import RAGConfig
from src.caching import RAGCacheManager, compute_doc_hash, normalize_query, cosine_similarity
from src.metrics import MetricsTracker, calculate_llm_cost

class TestCachingAndMetrics(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.cache_db = os.path.join(self.test_dir, "test_cache.db")
        self.metrics_db = os.path.join(self.test_dir, "test_metrics.db")
        self.cache_manager = RAGCacheManager(db_path=self.cache_db)
        self.metrics_tracker = MetricsTracker(db_path=self.metrics_db)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_exact_cache_hit_and_miss(self):
        doc_hash = "hash_pdf_a"
        document_id = "pdf_a"
        query = "What is the main contribution of this paper?"
        model_name = "gemini-flash-lite-latest"
        prompt_version = "v1.0"
        answer = "The main contribution is hybrid search RAG."
        citations = [1, 2]
        sources = [{"page": 1, "chunk_id": "c1", "content": "Text 1"}]

        # Before put -> Miss
        hit = self.cache_manager.get_exact(doc_hash, query, model_name, prompt_version)
        self.assertIsNone(hit)

        # Write to cache
        self.cache_manager.put(
            doc_hash=doc_hash,
            document_id=document_id,
            query=query,
            query_embedding=[0.1, 0.2, 0.3],
            model_name=model_name,
            prompt_version=prompt_version,
            answer=answer,
            citations=citations,
            sources=sources
        )

        # 1. Same PDF + same query -> Exact Cache HIT
        hit = self.cache_manager.get_exact(doc_hash, query, model_name, prompt_version)
        self.assertIsNotNone(hit)
        self.assertEqual(hit["cache_type"], "exact")
        self.assertEqual(hit["answer"], answer)

        # 3. Same query but different PDF -> Cache MISS
        hit_diff_pdf = self.cache_manager.get_exact("hash_pdf_b", query, model_name, prompt_version)
        self.assertIsNone(hit_diff_pdf)

        # 6. Changed prompt version -> Cache MISS
        hit_diff_prompt = self.cache_manager.get_exact(doc_hash, query, model_name, "v2.0")
        self.assertIsNone(hit_diff_prompt)

    def test_semantic_cache_hit_and_miss(self):
        doc_hash = "hash_pdf_a"
        document_id = "pdf_a"
        model_name = "gemini-flash-lite-latest"
        prompt_version = "v1.0"
        answer = "The paper contributes hybrid search."

        # Query vector A (original)
        vec_a = [1.0, 0.0, 0.0]
        # Query vector B (paraphrased, cosine sim ~ 0.98 > 0.90)
        vec_b = [0.98, 0.19, 0.0]
        # Query vector C (unrelated, cosine sim ~ 0.0 < 0.90)
        vec_c = [0.0, 1.0, 0.0]

        self.cache_manager.put(
            doc_hash=doc_hash,
            document_id=document_id,
            query="What does this paper contribute?",
            query_embedding=vec_a,
            model_name=model_name,
            prompt_version=prompt_version,
            answer=answer,
            citations=[1],
            sources=[]
        )

        # 2. Same PDF + paraphrased query above threshold -> Semantic Cache HIT
        hit_sem = self.cache_manager.get_semantic(
            doc_hash=doc_hash,
            query="What is the main contribution of this paper?",
            query_embedding=vec_b,
            model_name=model_name,
            prompt_version=prompt_version,
            threshold=0.90
        )
        self.assertIsNotNone(hit_sem)
        self.assertEqual(hit_sem["cache_type"], "semantic")
        self.assertGreaterEqual(hit_sem["similarity"], 0.90)

        # 4. Similar query below threshold -> Semantic Cache MISS
        hit_unrelated = self.cache_manager.get_semantic(
            doc_hash=doc_hash,
            query="What is the author's contact address?",
            query_embedding=vec_c,
            model_name=model_name,
            prompt_version=prompt_version,
            threshold=0.90
        )
        self.assertIsNone(hit_unrelated)

        # 5. Changed document version -> Semantic Cache MISS
        hit_diff_doc = self.cache_manager.get_semantic(
            doc_hash="hash_pdf_updated",
            query="What is the main contribution?",
            query_embedding=vec_b,
            model_name=model_name,
            prompt_version=prompt_version,
            threshold=0.90
        )
        self.assertIsNone(hit_diff_doc)

    def test_cost_calculation(self):
        pricing = {
            "gemini-flash-lite-latest": {"input_per_1m": 0.075, "output_per_1m": 0.30}
        }
        # 1,000,000 input tokens = $0.075, 1,000,000 output tokens = $0.30 -> Total = $0.375
        cost = calculate_llm_cost("gemini-flash-lite-latest", 1_000_000, 1_000_000, pricing)
        self.assertAlmostEqual(cost, 0.375, places=5)

        # 2,000 input tokens, 500 output tokens
        cost_small = calculate_llm_cost("gemini-flash-lite-latest", 2000, 500, pricing)
        expected = (2000 / 1e6) * 0.075 + (500 / 1e6) * 0.30
        self.assertAlmostEqual(cost_small, round(expected, 6), places=5)

    def test_metrics_telemetry_logging(self):
        self.metrics_tracker.log_request(
            document_id="doc1",
            user_id="u1",
            model_name="gemini-flash-lite-latest",
            input_tokens=1000,
            output_tokens=200,
            latency_ms=1500.0,
            retrieval_latency_ms=200.0,
            llm_latency_ms=1300.0,
            computed_cost_usd=0.000135,
            cache_hit=False,
            cache_type="miss",
            request_category="pdf_qa",
            num_retrieved_chunks=5
        )

        self.metrics_tracker.log_request(
            document_id="doc1",
            user_id="u1",
            model_name="gemini-flash-lite-latest",
            input_tokens=0,
            output_tokens=0,
            latency_ms=15.0,
            retrieval_latency_ms=0.0,
            llm_latency_ms=0.0,
            computed_cost_usd=0.0,
            cache_hit=True,
            cache_type="exact",
            request_category="pdf_qa",
            num_retrieved_chunks=0
        )

        stats = self.metrics_tracker.get_summary_stats()
        self.assertEqual(stats["total_requests"], 2)
        self.assertEqual(stats["cache_hits"], 1)
        self.assertEqual(stats["exact_hits"], 1)
        self.assertEqual(stats["cache_misses"], 1)
        self.assertEqual(stats["cache_hit_rate"], 50.0)

        recent = self.metrics_tracker.get_recent_requests()
        self.assertEqual(len(recent), 2)

    def test_graceful_fault_tolerance(self):
        # 7 & 8. Verify that database connection errors do not raise uncaught exceptions
        invalid_path = os.path.join(self.test_dir, "invalid_file.txt", "impossible_dir", "cache.db")
        # Create invalid_file.txt as a file so os.makedirs fails
        with open(os.path.join(self.test_dir, "invalid_file.txt"), "w") as f:
            f.write("file_not_dir")

        broken_cache = RAGCacheManager(db_path=invalid_path)
        res = broken_cache.get_exact("h1", "q1", "m1", "v1")
        self.assertIsNone(res)

        broken_metrics = MetricsTracker(db_path=invalid_path)
        broken_metrics.log_request(
            document_id="d", user_id="u", model_name="m", input_tokens=0, output_tokens=0,
            latency_ms=0, retrieval_latency_ms=0, llm_latency_ms=0, computed_cost_usd=0,
            cache_hit=False, cache_type="miss"
        )
        stats = broken_metrics.get_summary_stats()
        self.assertEqual(stats["total_requests"], 0)

if __name__ == "__main__":
    unittest.main()
