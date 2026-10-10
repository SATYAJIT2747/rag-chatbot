import re
from typing import Any, Dict, List
from .models import ModelRegistry

MAX_HISTORY_TURNS = 3
MAX_ANSWER_CHARS = 500

CONDENSE_SYSTEM_PROMPT = """You rewrite a follow-up question from a conversation about a PDF document into ONE standalone question.

Rules:
- Use the conversation only to resolve references: pronouns (it, he, they, that), phrases like "the second one", and omitted subjects.
- Write a complete natural-language question, not a list of keywords.
- Keep every specific term, name and number from the follow-up.
- If the follow-up is already understandable on its own, or is unrelated to the conversation, return it unchanged.
- Do NOT answer the question. Output ONLY the standalone question."""


def same_question(a: str, b: str) -> bool:
    """True when two questions differ only in case, spacing or punctuation."""
    return re.sub(r"[^a-z0-9]+", "", a.lower()) == re.sub(r"[^a-z0-9]+", "", b.lower())


def recent_turns(messages: List[Dict[str, Any]], max_turns: int = MAX_HISTORY_TURNS) -> List[Dict[str, str]]:
    """Pairs the chat's user/assistant messages into the last few {question, answer} turns."""
    turns = []
    pending_question = None
    for msg in messages:
        if msg.get("role") == "user":
            # A follow-up is remembered in its resolved form, so later follow-ups can build on it
            pending_question = msg.get("standalone") or msg.get("content", "")
        elif msg.get("role") == "assistant" and pending_question:
            turns.append({"question": pending_question, "answer": msg.get("content", "")})
            pending_question = None
    return turns[-max_turns:]


def condense_question(question: str, turns: List[Dict[str, str]], registry: ModelRegistry) -> str:
    """Session memory: rewrites a follow-up into a standalone question using the recent turns.

    Returns the question unchanged when there is no history, so a first question
    costs no LLM call and runs through the pipeline exactly as before.
    """
    if not turns or not question.strip():
        return question

    llm, tokenizer = registry.load_llm()
    # Only chat-style clients (Gemini, llama-server) are supported; HF transformers models keep the raw question
    if not hasattr(llm, "invoke") or tokenizer is not None:
        return question

    conversation = "\n".join(
        f"User: {t['question']}\nAssistant: {t['answer'][:MAX_ANSWER_CHARS]}" for t in turns
    )
    messages = [
        ("system", CONDENSE_SYSTEM_PROMPT),
        ("user", f"Conversation:\n{conversation}\n\nFollow-up question: {question}\nStandalone question:")
    ]
    try:
        response = llm.invoke(messages)
    except Exception as e:
        print(f"[Memory] Warning: follow-up rewrite failed, using the question as typed - {e}")
        return question

    content = response.content if hasattr(response, "content") else str(response)
    if isinstance(content, list):
        content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    lines = [line.strip() for line in str(content).strip().splitlines() if line.strip()]
    standalone = lines[0].strip('"').strip() if lines else ""

    # An empty or runaway rewrite is worse than no rewrite
    if not standalone or len(standalone) > 3 * len(question) + 300:
        return question
    return standalone
