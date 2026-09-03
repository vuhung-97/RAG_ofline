"""Context builder — SRP: chỉ build formatted context cho LLM."""

from typing import List, Dict, Any


def build_context(
    chunks: List[Dict[str, Any]],
    max_chunks: int = 8,
    max_tokens: int = 3000
) -> str:
    """
    Build formatted context string for LLM.
    Respects both chunk count and token budget.
    """
    context_parts = []
    total_tokens = 0

    for idx, chunk in enumerate(chunks[:max_chunks], 1):
        text = chunk.get("text", "")
        meta = chunk.get("metadata", {})

        est_tokens = len(text) // 4
        if total_tokens + est_tokens > max_tokens:
            # Try partial inclusion if we haven't reached minimum
            remaining = max_tokens - total_tokens
            if remaining > 200:
                text = text[:remaining * 4]
                est_tokens = remaining
            else:
                break

        header = f"[{idx}]"
        source_parts = []
        doc_name = meta.get("file_name", "")
        page = meta.get("page", "")
        chapter = meta.get("chapter", "")
        heading = meta.get("heading", "")

        if doc_name:
            source_parts.append(f"Tài liệu: {doc_name}")
        if page:
            source_parts.append(f"Trang: {page}")
        if chapter:
            source_parts.append(f"Chương: {chapter}")
        if heading:
            source_parts.append(f"Mục: {heading}")

        source_info = " | ".join(source_parts)
        if source_info:
            header += f" ({source_info})"

        context_parts.append(f"{header}\n{text}")
        total_tokens += est_tokens

    return "\n\n".join(context_parts)


def format_sources_for_ui(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Format chunks thành list sources cho UI display."""
    sources = []
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        sources.append({
            "chunk_id": chunk.get("chunk_id", ""),
            "file_name": meta.get("file_name", ""),
            "chapter": meta.get("chapter", ""),
            "heading": meta.get("heading", ""),
            "page": meta.get("page", ""),
            "text": chunk.get("text", "")[:200],
            "rrf_score": chunk.get("rrf_score", 0.0),
            "semantic_score": chunk.get("semantic_score", 0.0),
            "bm25_score": chunk.get("bm25_score", 0.0),
        })
    return sources
