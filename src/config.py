import os
import torch
import psutil
from dataclasses import dataclass
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Automatically look for .env in current directory or parent directory
load_dotenv()
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

@dataclass
class RAGConfig:
    # Target PDF path (searches data/ folder, current directory, or parent directory)
    pdf_path: str = os.getenv("RAG_PDF_PATH", "data/sample_document.pdf" if os.path.exists("data/sample_document.pdf") else ("sample_document.pdf" if os.path.exists("sample_document.pdf") else "data/sample_document.pdf"))
    artifacts_dir: str = "artifacts"
    faiss_dir: str = "artifacts/faiss"
    bm25_path: str = "artifacts/bm25/bm25.pkl"
    report_dir: str = "artifacts/report"
    
    # Model Identifiers
    embedding_model_id: str = "BAAI/bge-small-en-v1.5"
    reranker_model_id: str = "BAAI/bge-reranker-base"
    llm_model_id: str = "gemini-flash-lite-latest"
    
    # Chunking Configuration
    chunk_size: int = 512
    chunk_overlap: int = 50
    
    # Retrieval Hyperparameters
    retrieval_k: int = 20
    rerank_k: int = 10
    final_k: int = 5
    rrf_k: int = 60
    
    # Generation & Context
    return_parent_context: bool = True
    max_parent_context_chars: int = 4000
    llm_max_new_tokens: int = 512
    llm_temperature: float = 0.1
    
    # Hardware & Precision
    use_4bit_quantization: bool = False
    device_override: Optional[str] = None

def get_system_device_info(config: Optional[RAGConfig] = None) -> Dict[str, Any]:
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device": "cpu",
        "gpu_name": "None",
        "vram_gb": 0.0,
        "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2)
    }
    
    if config and config.device_override:
        info["device"] = config.device_override
    elif info["cuda_available"]:
        info["device"] = "cuda"
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
    
    return info
