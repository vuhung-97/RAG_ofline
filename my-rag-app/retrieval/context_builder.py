"""Context builder — SRP: chỉ build formatted context cho LLM."""

from typing import List, Dict, Any
from config import config


def _merge_table_chunks(chunks: List[Dict[str, Any]], max_chars: int = None) -> List[Dict[str, Any]]:
    """Gom các chunks có cùng table_id thành chunk duy nhất.

    Merge 2 chiều: chỉ cần có table_id → tìm tất cả chunks cùng table_id → merge.
    Truncate tại row boundary nếu vượt max_chars.

    Args:
        chunks: Danh sách chunks (có thể chứa non-table và table chunks).
        max_chars: Số ký tự tối đa cho 1 bảng. Mặc định dùng config.TABLE_MAX_CHARS.

    Returns:
        Danh sách chunks đã merge.
    """
    if max_chars is None:
        max_chars = config.TABLE_MAX_CHARS

    # Group chunks theo table_id
    table_groups: Dict[int, List[Dict]] = {}
    non_table_chunks = []

    for chunk in chunks:
        tid = chunk.get("metadata", {}).get("table_id")
        if tid is not None:
            table_groups.setdefault(tid, []).append(chunk)
        else:
            non_table_chunks.append(chunk)

    # Merge mỗi group theo chunk_index
    merged_tables = []
    for tid, group in table_groups.items():
        # Sort theo chunk_index để đảm bảo thứ tự đúng
        group.sort(key=lambda c: c["metadata"].get("chunk_index", 0))

        # Merge text
        merged_text = "\n".join(c.get("text", "") for c in group)

        # Truncate nếu vượt max_chars (cutoff tại row boundary)
        if len(merged_text) > max_chars:
            lines = merged_text.split("\n")
            truncated = []
            current_len = 0
            for line in lines:
                if current_len + len(line) + 1 > max_chars:
                    break
                truncated.append(line)
                current_len += len(line) + 1
            merged_text = "\n".join(truncated)

        # Metadata: lấy từ chunk đầu tiên (table_start hoặc chunk có table_id đầu tiên)
        merged_meta = group[0]["metadata"].copy()
        merged_meta["type"] = "table"
        merged_meta["table_id"] = tid
        merged_meta["chunk_count"] = len(group)

        merged_tables.append({
            "text": merged_text,
            "metadata": merged_meta,
            "chunk_id": group[0].get("chunk_id", ""),
            # Giữ scores cao nhất trong group
            "rrf_score": max(c.get("rrf_score", 0) for c in group),
            "semantic_score": max(c.get("semantic_score", 0) for c in group),
            "bm25_score": max(c.get("bm25_score", 0) for c in group),
        })

    # Kết quả: non-table chunks + merged table chunks
    return non_table_chunks + merged_tables


def build_context(
    chunks: List[Dict[str, Any]],
    max_chunks: int = 8,
    max_tokens: int = 3000
) -> tuple:
    """
    Build formatted context string for LLM.
    Respects both chunk count and token budget.

    Returns:
        Tuple[str, List[Dict]]: (formatted_context, merged_chunks)
    """
    # Merge table chunks trước khi build context
    merged = _merge_table_chunks(chunks)

    context_parts = []
    total_tokens = 0
    used_chunks = []

    for idx, chunk in enumerate(merged[:max_chunks], 1):
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
        used_chunks.append(chunk)

    return "\n\n".join(context_parts), used_chunks


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
