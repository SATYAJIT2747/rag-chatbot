import re
from typing import Dict, Any, List, Optional, Tuple

INJECTION_PATTERNS = [
    r'ignore\s+(all\s+)?(previous|prior|above|system)\s+instructions',
    r'forget\s+(your|all|previous|prior)\s+instructions',
    r'(reveal|show|display|print|echo|output|tell\s+me)\s+(your|the)?\s*system\s+prompt',
    r'ignore\s+(the\s+)?system\s+prompt',
    r'you\s+are\s+now\s+(unrestricted|jailbroken|dan|godmode)',
    r'override\s+(system|safety|security)\s+rules',
    r'disregard\s+(all\s+)?(safety\s+)?(rules|instructions)',
    r'bypass\s+(all\s+)?safety',
    r'act\s+as\s+an\s+unfiltered',
    r'new\s+system\s+instructions:'
]

def validate_query_length(query: str, max_length: int = 2000) -> Tuple[bool, Optional[str]]:
    if not query or not query.strip():
        return False, "Query cannot be empty."
    if len(query) > max_length:
        return False, "Your question is too long. Please ask a shorter question."
    return True, None

def detect_prompt_injection(query: str) -> Tuple[bool, Optional[str]]:
    norm_q = query.lower().strip()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, norm_q):
            return True, "I can't process requests that attempt to override the assistant's instructions."
    return False, None

def validate_pdf_scope(
    final_docs: List[Any],
    debug_payload: Optional[Dict[str, Any]] = None,
    threshold: float = -4.0
) -> Tuple[bool, Optional[str]]:
    if not final_docs:
        return False, "I can only answer questions related to the uploaded PDF."
    
    if debug_payload and isinstance(debug_payload, dict) and "rerank_debug" in debug_payload:
        rerank_items = debug_payload.get("rerank_debug", [])
        if isinstance(rerank_items, list) and rerank_items:
            scores = []
            for item in rerank_items:
                if isinstance(item, dict):
                    score = item.get("rerank_score") if item.get("rerank_score") is not None else item.get("score")
                    if score is not None:
                        try:
                            scores.append(float(score))
                        except (ValueError, TypeError):
                            pass
            if scores and max(scores) < threshold:
                return False, "I can only answer questions related to the uploaded PDF."
                
    return True, None

def validate_output_grounding(
    query: str,
    context: str,
    answer: str
) -> Dict[str, Any]:
    if not answer or not answer.strip():
        return {"grounded": False, "reason": "Empty answer generated", "confidence": 0.0}
        
    refusal_phrase = "the provided document does not contain enough information"
    if refusal_phrase in answer.lower():
        return {"grounded": True, "reason": "Standard safe refusal response", "confidence": 1.0}
        
    # Extract numbers or quantities from answer
    answer_numbers = set(re.findall(r'\b\d+(?:,\d+)*(?:\.\d+)?\b', answer))
    context_numbers = set(re.findall(r'\b\d+(?:,\d+)*(?:\.\d+)?\b', context))
    
    # Check if any fabricated numbers exist in answer that are absent in context
    unsupported_numbers = answer_numbers - context_numbers
    if unsupported_numbers:
        # If numbers like page citations [Page X] are present, filter out valid citation page numbers
        citations = set(re.findall(r'\[Pages?\s+([0-9,\s]+)\]', answer, flags=re.IGNORECASE))
        cit_nums = set()
        for c in citations:
            for n in re.findall(r'\d+', c):
                cit_nums.add(n)
        unsupported_numbers = unsupported_numbers - cit_nums
        
        if unsupported_numbers:
            return {
                "grounded": False,
                "reason": f"Factual number mismatch: {list(unsupported_numbers)} not found in retrieved PDF context",
                "confidence": 0.2
            }

    # Keyword overlap ratio check between answer non-stop-words and context
    words = [w.lower() for w in re.findall(r'\b[a-zA-Z]{3,}\b', answer)]
    stop_words = {"the", "and", "is", "in", "it", "of", "to", "for", "with", "on", "that", "this", "are", "by", "from", "as", "at", "an", "be", "has", "have", "not", "or", "which", "was", "does", "based", "provided", "document"}
    meaningful_words = [w for w in words if w not in stop_words]
    
    if meaningful_words:
        context_lower = context.lower()
        matched = [w for w in meaningful_words if w in context_lower]
        overlap_ratio = len(matched) / len(meaningful_words)
        
        if overlap_ratio < 0.30:
            return {
                "grounded": False,
                "reason": f"Low textual grounding overlap ({overlap_ratio:.2f} < 0.30)",
                "confidence": round(overlap_ratio, 2)
            }
            
    return {"grounded": True, "reason": "Answer claims are supported by context", "confidence": 0.95}

class RAGGuardrailsManager:
    def __init__(self, config: Any):
        self.config = config

    def check_input(self, query: str) -> Tuple[bool, Optional[str], str]:
        max_len = getattr(self.config, "max_query_length", 2000)
        valid_len, len_msg = validate_query_length(query, max_length=max_len)
        if not valid_len:
            return False, len_msg, "blocked_length"
            
        enable_inj = getattr(self.config, "enable_prompt_injection_guard", True)
        if enable_inj:
            is_inj, inj_msg = detect_prompt_injection(query)
            if is_inj:
                return False, inj_msg, "blocked_injection"
                
        return True, None, "allowed"

    def check_scope(self, final_docs: List[Any], debug_payload: Optional[Dict[str, Any]] = None) -> Tuple[bool, Optional[str]]:
        enable_scope = getattr(self.config, "enable_scope_guard", True)
        if not enable_scope:
            return True, None
            
        threshold = getattr(self.config, "pdf_scope_threshold", -4.0)
        return validate_pdf_scope(final_docs, debug_payload=debug_payload, threshold=threshold)

    def check_output_grounding(self, query: str, context: str, answer: str) -> Dict[str, Any]:
        enable_grounding = getattr(self.config, "enable_grounding_guard", True)
        if not enable_grounding:
            return {"grounded": True, "reason": "Grounding check disabled", "confidence": 1.0}
            
        return validate_output_grounding(query, context, answer)
