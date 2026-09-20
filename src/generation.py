import re
import time
import torch
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from .models import ModelRegistry

def build_context(
    retrieved_chunks: List[Document],
    page_map: Dict[int, str],
    return_parent_context: bool = True,
    max_context_chars: int = 4000
) -> str:
    context_blocks = []
    seen_pages = set()
    total_chars = 0
    
    if return_parent_context:
        for chunk in retrieved_chunks:
            page_num = chunk.metadata["page"]
            if page_num in seen_pages:
                continue
            seen_pages.add(page_num)
            page_text = page_map.get(page_num, chunk.page_content)
            
            block = f"--- [Page {page_num}] (Source: {chunk.metadata['source']}) ---\n{page_text}\n"
            if total_chars + len(block) > max_context_chars and len(context_blocks) > 0:
                break
            context_blocks.append(block)
            total_chars += len(block)
    else:
        for chunk in retrieved_chunks:
            page_num = chunk.metadata["page"]
            block = f"--- [Page {page_num}] (Chunk: {chunk.metadata.get('chunk_id', 'unknown')}) ---\n{chunk.page_content}\n"
            if total_chars + len(block) > max_context_chars and len(context_blocks) > 0:
                break
            context_blocks.append(block)
            total_chars += len(block)
            
    return "\n".join(context_blocks)

def generate_answer(
    query: str,
    context: str,
    registry: ModelRegistry,
    max_new_tokens: int = 512,
    temperature: float = 0.1
) -> Dict[str, Any]:
    t0 = time.time()
    llm, tokenizer = registry.load_llm()
    
    system_prompt = """You are a document question-answering assistant.
Answer the user's question using ONLY the retrieved context.
If the retrieved context does not contain enough information to answer the question, output EXACTLY:
"The provided document does not contain enough information to answer this."

Do NOT invent facts. Every factual claim must be supported by the retrieved context.
Include page citations in your answer using the format [Page X] or [Pages X, Y] matching the page headers in the context.
Do not fabricate page numbers."""

    user_message = f"""Context:
{context}

Question: {query}

Answer:"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]
    
    if hasattr(llm, "invoke") and tokenizer is None:
        messages = [
            ("system", system_prompt),
            ("user", user_message)
        ]
        response = llm.invoke(messages)
        content = response.content if hasattr(response, "content") else str(response)
        if isinstance(content, list):
            answer = "".join([part.get("text", "") if isinstance(part, dict) else str(part) for part in content]).strip()
        else:
            answer = str(content).strip()
    else:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(registry.device)
        
        with torch.no_grad():
            output_ids = llm.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=False if temperature == 0.0 else True,
                pad_token_id=tokenizer.eos_token_id
            )
            
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        
    latency_sec = time.time() - t0
    
    return {
        "answer": answer,
        "latency_sec": round(latency_sec, 3)
    }

def extract_citations(answer_text: str) -> List[int]:
    citations = set()
    matches = re.findall(r'\[Pages?\s+([0-9,\s]+)\]', answer_text, flags=re.IGNORECASE)
    for m in matches:
        nums = re.findall(r'\d+', m)
        for n in nums:
            citations.add(int(n))
    return sorted(list(citations))

def format_cited_response(
    query: str,
    rewritten_query: Optional[str],
    answer: str,
    retrieved_docs: List[Document],
    latency_breakdown: Dict[str, float],
    debug_payload: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    citations = extract_citations(answer)
    sources = []
    for doc in retrieved_docs:
        sources.append({
            "page": doc.metadata.get("page", 0),
            "source": doc.metadata.get("source", "unknown"),
            "chunk_id": doc.metadata.get("chunk_id", "unknown"),
            "content": doc.page_content
        })
        
    return {
        "query": query,
        "rewritten_query": rewritten_query,
        "answer": answer,
        "citations": citations,
        "sources": sources,
        "latency_breakdown": latency_breakdown,
        "debug_payload": debug_payload or {}
    }
