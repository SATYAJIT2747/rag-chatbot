import os
import re
import httpx
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Union

THINK_BLOCK = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)


@dataclass
class LocalChatResponse:
    """Mirrors the fields of a LangChain AIMessage that src/generation.py reads."""
    content: str
    response_metadata: Dict[str, Any] = field(default_factory=dict)


class LlamaServerChat:
    """
    Minimal chat client for llama.cpp's OpenAI-compatible server (llama-server).

    Exposes `.invoke(messages)` so it can be dropped into ModelRegistry.llm_model
    and go through the same generation / query-rewrite code path as Gemini.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080/v1",
        model: str = "qwen3.5-4b-q8",
        temperature: float = 0.1,
        max_tokens: int = 512,
        enable_thinking: bool = False,
        timeout_sec: float = 300.0
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        self.client = httpx.Client(timeout=timeout_sec)

    def served_model(self) -> str:
        """Returns the GGUF path llama-server currently has loaded (useful to confirm the quant)."""
        resp = self.client.get(f"{self.base_url}/models")
        resp.raise_for_status()
        data = resp.json().get("data", [])
        return data[0]["id"] if data else "unknown"

    def invoke(self, messages: List[Union[Tuple[str, str], Dict[str, str]]]) -> LocalChatResponse:
        chat_messages = []
        for m in messages:
            if isinstance(m, tuple):
                role, content = m
            else:
                role, content = m["role"], m["content"]
            chat_messages.append({"role": "assistant" if role == "ai" else role, "content": content})

        payload = {
            "model": self.model,
            "messages": chat_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            # Qwen3.5 is a hybrid reasoning model; thinking is off by default for RAG answers
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking}
        }
        resp = self.client.post(f"{self.base_url}/chat/completions", json=payload)
        resp.raise_for_status()
        body = resp.json()

        message = body["choices"][0]["message"]
        answer = THINK_BLOCK.sub("", message.get("content") or "").strip()
        usage = body.get("usage", {})
        return LocalChatResponse(
            content=answer,
            response_metadata={
                "token_usage": {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0)
                },
                "timings": body.get("timings", {}),
                "finish_reason": body["choices"][0].get("finish_reason")
            }
        )


def client_from_env(temperature: float = 0.1, max_tokens: int = 512) -> LlamaServerChat:
    return LlamaServerChat(
        base_url=os.getenv("LOCAL_LLM_BASE_URL", "http://127.0.0.1:8080/v1"),
        model=os.getenv("LOCAL_LLM_MODEL", "qwen3.5-4b-q8"),
        temperature=temperature,
        max_tokens=max_tokens,
        enable_thinking=os.getenv("LOCAL_LLM_THINKING", "false").lower() == "true"
    )
