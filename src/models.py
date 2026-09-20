import os
import time
import torch
import warnings
from typing import Tuple, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from .config import RAGConfig

# Suppress benign huggingface warnings on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)

class ModelRegistry:
    def __init__(self, config: RAGConfig, device: str = "cpu"):
        self.config = config
        self.device = device
        self.embedder = None
        self.reranker_model = None
        self.reranker_tokenizer = None
        self.llm_model = None
        self.llm_tokenizer = None

    def load_embedding_model(self) -> SentenceTransformer:
        if self.embedder is None:
            t0 = time.time()
            self.embedder = SentenceTransformer(self.config.embedding_model_id, device=self.device)
            print(f"[ModelRegistry] Loaded Embeddings: {self.config.embedding_model_id} in {time.time()-t0:.2f}s")
        return self.embedder

    def load_reranker_model(self) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
        if self.reranker_model is None:
            t0 = time.time()
            self.reranker_tokenizer = AutoTokenizer.from_pretrained(self.config.reranker_model_id)
            model_dtype = torch.float16 if self.device == "cuda" else torch.float32
            self.reranker_model = AutoModelForSequenceClassification.from_pretrained(
                self.config.reranker_model_id,
                dtype=model_dtype
            ).to(self.device)
            self.reranker_model.eval()
            print(f"[ModelRegistry] Loaded Reranker: {self.config.reranker_model_id} in {time.time()-t0:.2f}s")
        return self.reranker_model, self.reranker_tokenizer

    def load_llm(self) -> Tuple[Any, Optional[AutoTokenizer]]:
        if self.llm_model is None:
            t0 = time.time()
            if "gemini" in self.config.llm_model_id.lower():
                from langchain_google_genai import ChatGoogleGenerativeAI
                api_key = os.getenv("GOOGLE_API_KEY")
                self.llm_model = ChatGoogleGenerativeAI(
                    model=self.config.llm_model_id,
                    google_api_key=api_key,
                    temperature=self.config.llm_temperature,
                    max_output_tokens=self.config.llm_max_new_tokens
                )
                self.llm_tokenizer = None
                print(f"[ModelRegistry] Loaded Gemini LLM: {self.config.llm_model_id} in {time.time()-t0:.2f}s", flush=True)
            else:
                self.llm_tokenizer = AutoTokenizer.from_pretrained(self.config.llm_model_id)
                model_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

                if self.config.use_4bit_quantization and self.device == "cuda":
                    from transformers import BitsAndBytesConfig
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True
                    )
                    self.llm_model = AutoModelForCausalLM.from_pretrained(
                        self.config.llm_model_id,
                        quantization_config=bnb_config,
                        device_map={"": self.device}
                    )
                else:
                    self.llm_model = AutoModelForCausalLM.from_pretrained(
                        self.config.llm_model_id,
                        dtype=model_dtype,
                        device_map={"": self.device} if self.device == "cuda" else None
                    )
                self.llm_model.eval()
                print(f"[ModelRegistry] Loaded Local LLM: {self.config.llm_model_id} in {time.time()-t0:.2f}s", flush=True)
        return self.llm_model, self.llm_tokenizer
