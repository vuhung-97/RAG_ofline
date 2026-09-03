"""Neighbor expansion — SRP: merge chunk lân cận vào chunk gốc."""

from typing import List, Dict, Any


def expand_neighbors(
    results: List[Dict[str, Any]],
    all_chunks_map: Dict[str, Dict[str, Any]],
    max_expansion: int = 1,
    same_section_only: bool = True
) -> List[Dict[str, Any]]:
    """
    Merge chunk lân cận (previous/next) VÀO chunk gốc thay vì tạo chunk mới.

    Kết quả: số chunks giữ nguyên = top_k, nhưng mỗi chunk giàu nội dung hơn.

    Args:
        results: retrieved chunks (đã dedup, sorted by relevance)
        all_chunks_map: {chunk_id: {text, metadata, prev_id, next_id}}
        max_expansion: max chunks thêm mỗi bên (per result)
        same_section_only: chỉ thêm nếu cùng section/chapter
    """
    expanded_results = []

    for result in results:
        chunk_id = result.get("chunk_id", "")
        chunk_info = all_chunks_map.get(chunk_id, {})
        if not chunk_info:
            expanded_results.append(result)
            continue

        doc_name = result.get("metadata", {}).get("file_name", "")
        section = result.get("metadata", {}).get("heading", "")

        # Lấy prev chunks
        prev_texts = []
        prev_id = chunk_info.get("prev_id")
        added_prev = 0
        while prev_id and added_prev < max_expansion:
            prev_chunk = all_chunks_map.get(prev_id)
            if not prev_chunk:
                break
            if same_section_only:
                prev_section = prev_chunk.get("metadata", {}).get("heading", "")
                prev_doc = prev_chunk.get("metadata", {}).get("file_name", "")
                if prev_doc != doc_name or prev_section != section:
                    break
            prev_texts.insert(0, prev_chunk["text"])
            prev_id = prev_chunk.get("prev_id")
            added_prev += 1

        # Lấy next chunks
        next_texts = []
        next_id = chunk_info.get("next_id")
        added_next = 0
        while next_id and added_next < max_expansion:
            next_chunk = all_chunks_map.get(next_id)
            if not next_chunk:
                break
            if same_section_only:
                next_section = next_chunk.get("metadata", {}).get("heading", "")
                next_doc = next_chunk.get("metadata", {}).get("file_name", "")
                if next_doc != doc_name or next_section != section:
                    break
            next_texts.append(next_chunk["text"])
            next_id = next_chunk.get("next_id")
            added_next += 1

        # Merge: prev + gốc + next
        merged_text = "\n".join(prev_texts + [result["text"]] + next_texts)

        expanded_results.append({
            **result,
            "text": merged_text,
        })

    return expanded_results
