import os
import re
import json
from typing import List, Dict, Any, Tuple
import pymupdf as fitz
from langchain_core.documents import Document

def decode_char(c: str) -> str:
    """Decodes custom Private Use Area (PUA) font ligatures found in academic slides/papers."""
    code = ord(c)
    if 0xe047 <= code <= 0xe060:
        return chr(code - 0xe000 + 26)
    elif 0xe02b <= code <= 0xe044:
        return chr(code - 0xe02b + ord('A'))
    elif 0xe001 <= code <= 0xe01a:
        return chr(code - 0xe000 + 64)
    elif code == 0xfb01 or code == 0xe005:
        return 'fi'
    elif code == 0xfb02:
        return 'fl'
    elif code == 0x000c or code == 0xfb00 or code == 0xe006:
        return 'ff'
    elif code == 0xe008:
        return 'ffi'
    elif code == 0xe009:
        return 'ffl'
    elif code == 0x2212 or code == 0x2013 or code == 0x2014:
        return '-'
    elif 0xe000 <= code <= 0xf8ff:
        return ''
    return c

def decode_custom_ligatures(text: str) -> str:
    return "".join(decode_char(c) for c in text)

def extract_clean_page_text(page: fitz.Page) -> str:
    """Extracts text using bounding-box spatial gaps to fix merged tokens."""
    try:
        page_dict = json.loads(page.get_text("rawjson"))
        blocks_text = []
        for block in page_dict.get("blocks", []):
            if "lines" not in block:
                continue
            line_texts = []
            for line in block["lines"]:
                spans_chars = []
                for span in line.get("spans", []):
                    font_size = span.get("size", 12.0)
                    space_threshold = max(2.0, font_size * 0.20)
                    prev_x1 = None
                    for ch_info in span.get("chars", []):
                        c = ch_info.get("c", "")
                        bbox = ch_info.get("bbox", [0, 0, 0, 0])
                        x0, x1 = bbox[0], bbox[2]
                        if prev_x1 is not None and (x0 - prev_x1) > space_threshold:
                            if not spans_chars or spans_chars[-1] != ' ':
                                spans_chars.append(' ')
                        decoded = decode_char(c)
                        if decoded:
                            spans_chars.append(decoded)
                        prev_x1 = x1
                line_str = "".join(spans_chars).strip()
                if line_str:
                    line_texts.append(line_str)
            if line_texts:
                blocks_text.append("\n".join(line_texts))
        extracted = "\n\n".join(blocks_text).strip()
        if len(extracted) > 30:
            return extracted
    except Exception:
        pass
    raw = page.get_text("text").strip()
    return decode_custom_ligatures(raw)

def ensure_sample_pdf(filepath: str):
    """Checks for the PDF file; if absent, generates a default educational PDF."""
    if os.path.exists(filepath):
        return
    parent_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sample_document.pdf")
    if os.path.exists(parent_path):
        import shutil
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        shutil.copyfile(parent_path, filepath)
        return
        
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    doc = fitz.open()
    
    # Slide 1
    p1 = doc.new_page()
    p1.insert_text((50, 60), """ADVANCED RAG: FOUNDATIONS & HYBRID RETRIEVAL
Module 1: The Problem with Basic Vector Search

1. Limitations of Vanilla Semantic Search
Basic RAG systems rely exclusively on dense bi-encoder embeddings (e.g. cosine distance). While semantic search excels at abstract concept matching, it frequently fails on exact keywords, part numbers, abbreviations, acronyms, and rare technical terminology. Furthermore, top-k vector similarity is susceptible to embedding distribution collapse.

2. The Need for Hybrid Search
Hybrid retrieval bridges this gap by executing two complementary search mechanisms simultaneously:
- Sparse Keyword Search (BM25): Uses term-frequency / inverse-document-frequency with exact token matching.
- Dense Vector Search (FAISS + Bi-Encoder): Understands semantic context and synonym relationships.
Combining both systems guarantees high recall across both conceptual inquiries and exact keyword queries.""", fontsize=10.5)

    # Slide 2
    p2 = doc.new_page()
    p2.insert_text((50, 60), """ADVANCED RAG: FUSION & RERANKING
Module 2: Reciprocal Rank Fusion & Cross-Encoder Reranking

3. Reciprocal Rank Fusion (RRF)
RRF combines the ranked results of multiple independent retrieval systems without requiring raw score normalization. For each document d in rank list L:
    RRF_Score(d) = sum( 1 / (k + rank_i(d)) )
where k is a smoothing constant (typically k=60). Documents retrieved near the top of both lists receive the highest fused scores.

4. Cross-Encoder Reranking (bge-reranker-base)
Bi-encoders compute query and document representations independently in isolated vector spaces. In contrast, a Cross-Encoder performs full cross-attention over all (query, document) token pairs simultaneously:
    Score = CrossEncoder(Query, Document)
This captures deep semantic nuance and cross-term interactions, drastically eliminating false-positive candidates before LLM prompting.""", fontsize=10.5)

    # Slide 3
    p3 = doc.new_page()
    p3.insert_text((50, 60), """ADVANCED RAG: CONTEXT & CITATIONS
Module 3: Parent-Child Chunking & Strict Grounded Generation

5. Parent-Child Chunking & Context Expansion
Small chunks (e.g., 256–512 characters) optimize embedding precision and prevent vector dilution. However, LLMs require broader context to synthesize complete, nuanced answers. Parent-child context expansion indexes small child chunks for high-precision retrieval, but injects the full parent section / page into the final generation prompt.

6. Grounded Answering & Refusal Policy
To eliminate hallucinations, the generation prompt enforces strict citation attribution [Page X]. If the retrieved context is insufficient to answer the user query, the system deterministically outputs:
"The provided document does not contain enough information to answer this.\"""", fontsize=10.5)

    doc.save(filepath)
    doc.close()

def load_pdf(pdf_path: str) -> fitz.Document:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
    return fitz.open(pdf_path)

def extract_pages(doc: fitz.Document, source_name: str) -> List[Dict[str, Any]]:
    pages_data = []
    doc_id = os.path.splitext(os.path.basename(source_name))[0]
    
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        decoded_text = extract_clean_page_text(page)
        image_list = page.get_images(full=True)
        
        pages_data.append({
            "source": os.path.basename(source_name),
            "page": page_idx + 1,
            "document_id": doc_id,
            "text": decoded_text,
            "image_count": len(image_list),
            "content_type": "text" if len(decoded_text) > 50 else "scanned_or_sparse"
        })
    return pages_data

def ocr_page(page_data: Dict[str, Any], doc: fitz.Document) -> Dict[str, Any]:
    """Applies multimodal vision OCR if page text is missing or sparse."""
    if page_data.get("content_type") != "scanned_or_sparse" and len(page_data.get("text", "")) > 30:
        page_data["extraction_method"] = "text"
        return page_data
        
    page_num = page_data["page"]
    fitz_page = doc[page_num - 1]
    
    # 1. Primary OCR: Multimodal Vision OCR via Google GenAI (handles slides, charts, formulas with high fidelity)
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        try:
            import base64
            from langchain_google_genai import ChatGoogleGenerativeAI
            from langchain_core.messages import HumanMessage
            
            pix = fitz_page.get_pixmap(dpi=150)
            img_bytes = pix.tobytes("png")
            b64_img = base64.b64encode(img_bytes).decode("utf-8")
            
            ocr_llm = ChatGoogleGenerativeAI(
                model="gemini-flash-lite-latest",
                google_api_key=api_key,
                temperature=0.0
            )
            
            msg = HumanMessage(
                content=[
                    {"type": "text", "text": "Extract all text, headings, formulas, tables, code snippets, and bullet points from this lecture slide image verbatim into clean markdown format without commentary."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}}
                ]
            )
            res = ocr_llm.invoke([msg])
            content = res.content
            if isinstance(content, list):
                extracted = "".join([part.get("text", "") if isinstance(part, dict) else str(part) for part in content]).strip()
            else:
                extracted = str(content).strip()
                
            if len(extracted) > 15:
                page_data["text"] = extracted
                page_data["content_type"] = "ocr"
                page_data["extraction_method"] = "ocr"
                return page_data
        except Exception:
            pass

    # 2. Local OCR Fallback: PyTesseract / RapidOCR (if installed)
    try:
        import pytesseract
        from PIL import Image
        import io
        
        pix = fitz_page.get_pixmap(dpi=150)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        ocr_text = pytesseract.image_to_string(img).strip()
        
        if len(ocr_text) > 20:
            page_data["text"] = ocr_text
            page_data["content_type"] = "ocr"
            page_data["extraction_method"] = "ocr"
            return page_data
    except Exception:
        pass
        
    page_data["content_type"] = "ocr_skipped"
    page_data["extraction_method"] = "text"
    if len(page_data.get("text", "")) == 0:
        page_data["text"] = f"[Page {page_num}: Scanned or image content.]"
    return page_data

def clean_text(text: str) -> str:
    text = decode_custom_ligatures(text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    lines = [line.rstrip() for line in text.splitlines()]
    return '\n'.join(lines).strip()

def build_documents(pages_data: List[Dict[str, Any]]) -> List[Document]:
    documents = []
    for p in pages_data:
        cleaned = clean_text(p["text"])
        if not cleaned:
            continue
        doc = Document(
            page_content=cleaned,
            metadata={
                "source": p["source"],
                "page": p["page"],
                "document_id": p["document_id"],
                "extraction_method": p.get("extraction_method", "text"),
                "content_type": p.get("content_type", "text")
            }
        )
        documents.append(doc)
    return documents

def build_page_map(normalized_docs: List[Document]) -> Dict[int, str]:
    return {doc.metadata["page"]: doc.page_content for doc in normalized_docs}

def load_and_preprocess_pdf(pdf_path: str) -> Tuple[List[Document], Dict[int, str]]:
    ensure_sample_pdf(pdf_path)
    fitz_doc = load_pdf(pdf_path)
    pages_data = extract_pages(fitz_doc, pdf_path)
    processed_pages = [ocr_page(p, fitz_doc) for p in pages_data]
    docs = build_documents(processed_pages)
    page_map = build_page_map(docs)
    fitz_doc.close()
    return docs, page_map
