"""Neighbor expansion — SRP: chỉ mở rộng context từ chunk lân cận."""

from typing import List, Dict, Any


def expand_neighbors(
    results: List[Dict[str, Any]],
    all_chunks_map: Dict[str, Dict[str, Any]],
    max_expansion: int = 1,
    same_section_only: bool = True
) -> List[Dict[str, Any]]:
    """
    Mở rộng kết quả retrieval bằng chunk lân cận (previous/next).

    Args:
        results: retrieved chunks (đã dedup, sorted by relevance)
        all_chunks_map: {chunk_id: {text, metadata, prev_id, next_id}}
        max_expansion: max chunks thêm mỗi bên (per result)
        same_section_only: chỉ thêm nếu cùng section/chapter
    """
    expanded_ids = set()
    expanded_results = []

    for result in results:
        chunk_id = result.get("chunk_id", "")
        if not chunk_id or chunk_id in expanded_ids:
            continue

        expanded_results.append(result)
        expanded_ids.add(chunk_id)

        chunk_info = all_chunks_map.get(chunk_id, {})
        if not chunk_info:
            continue

        doc_name = result.get("metadata", {}).get("file_name", "")
        section = result.get("metadata", {}).get("heading", "")

        # Expand previous chunks
        prev_id = chunk_info.get("prev_id")
        added_prev = 0
        while prev_id and added_prev < max_expansion:
            if prev_id in expanded_ids:
                break
            prev_chunk = all_chunks_map.get(prev_id)
            if not prev_chunk:
                break
            if same_section_only:
                prev_section = prev_chunk.get("metadata", {}).get("heading", "")
                prev_doc = prev_chunk.get("metadata", {}).get("file_name", "")
                if prev_doc != doc_name or prev_section != section:
                    break
            prev_result = {
                "chunk_id": prev_id,
                "text": prev_chunk["text"],
                "metadata": prev_chunk["metadata"],
                "rrf_score": 0.0,
                "semantic_score": 0.0,
                "bm25_score": 0.0,
                "is_expanded": True
            }
            expanded_results.append(prev_result)
            expanded_ids.add(prev_id)
            prev_id = prev_chunk.get("prev_id")
            added_prev += 1

        # Expand next chunks
        next_id = chunk_info.get("next_id")
        added_next = 0
        while next_id and added_next < max_expansion:
            if next_id in expanded_ids:
                break
            next_chunk = all_chunks_map.get(next_id)
            if not next_chunk:
                break
            if same_section_only:
                next_section = next_chunk.get("metadata", {}).get("heading", "")
                next_doc = next_chunk.get("metadata", {}).get("file_name", "")
                if next_doc != doc_name or next_section != section:
                    break
            next_result = {
                "chunk_id": next_id,
                "text": next_chunk["text"],
                "metadata": next_chunk["metadata"],
                "rrf_score": 0.0,
                "semantic_score": 0.0,
                "bm25_score": 0.0,
                "is_expanded": True
            }
            expanded_results.append(next_result)
            expanded_ids.add(next_id)
            next_id = next_chunk.get("next_id")
            added_next += 1

    return expanded_results
