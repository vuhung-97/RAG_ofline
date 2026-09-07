"""Neighbor expansion — SRP: merge chunk lân cận + interval consolidation.

Sửa: Thêm interval consolidation để tránh duplicate khi nhiều chunks
expand cùng vùng lân cận.
"""

from typing import List, Dict, Any, Tuple


def expand_neighbors(
    results: List[Dict[str, Any]],
    all_chunks_map: Dict[str, Dict[str, Any]],
    max_expansion: int = 1,
    same_section_only: bool = True
) -> List[Dict[str, Any]]:
    """
    Merge chunk lân cận VÀO chunk gốc + consolidate overlapping intervals.

    Strategy:
    1. Thu thập tất cả chunk IDs cần mở rộng (original + neighbors)
    2. Gom thành continuous ranges (intervals) — merge overlapping
    3. Build context blocks từ consolidated ranges, deduplicate text

    Args:
        results: retrieved chunks (đã dedup, sorted by relevance)
        all_chunks_map: {chunk_id: {text, metadata, prev_id, next_id}}
        max_expansion: max chunks thêm mỗi bên (per result)
        same_section_only: chỉ thêm nếu cùng section/chapter
    """
    if not results:
        return []

    # Step 1: Collect all chunk IDs we want to expand
    all_ids_to_expand = set()
    original_id_set = set()

    for result in results:
        chunk_id = result.get("chunk_id", "")
        original_id_set.add(chunk_id)
        all_ids_to_expand.add(chunk_id)
        _collect_neighbor_ids(chunk_id, all_chunks_map, all_ids_to_expand,
                              max_expansion, same_section_only)

    # Step 2: Group into continuous ranges (intervals)
    intervals = _build_consolidated_intervals(all_ids_to_expand, all_chunks_map)

    # Step 3: Build expanded results from consolidated intervals
    result_ids = [r.get("chunk_id", "") for r in results]
    chunk_lookup = {r.get("chunk_id", ""): r for r in results}

    expanded_results = []
    used_ids = set()

    for result in results:
        chunk_id = result.get("chunk_id", "")
        if chunk_id in used_ids:
            continue

        # Find the interval containing this chunk
        interval_ids = _find_interval(chunk_id, intervals)
        if not interval_ids:
            interval_ids = [chunk_id]

        # Merge text from interval, dedup
        merged_texts = []
        for cid in interval_ids:
            if cid in used_ids:
                continue
            used_ids.add(cid)
            if cid in chunk_lookup:
                merged_texts.append(chunk_lookup[cid]["text"])
            elif cid in all_chunks_map:
                merged_texts.append(all_chunks_map[cid]["text"])

        merged_text = "\n".join(merged_texts) if merged_texts else result.get("text", "")

        expanded_results.append({
            **result,
            "text": merged_text,
        })

    return expanded_results


def _collect_neighbor_ids(
    chunk_id: str,
    all_chunks_map: Dict[str, Dict[str, Any]],
    collected: set,
    max_expansion: int,
    same_section_only: bool
):
    """Thu thập neighbor IDs để mở rộng."""
    chunk_info = all_chunks_map.get(chunk_id, {})
    if not chunk_info:
        return

    doc_name = chunk_info.get("metadata", {}).get("file_name", "")
    section = chunk_info.get("metadata", {}).get("heading", "")

    # Prev direction
    prev_id = chunk_info.get("prev_id")
    added = 0
    while prev_id and added < max_expansion:
        if prev_id in collected:
            break
        prev_chunk = all_chunks_map.get(prev_id)
        if not prev_chunk:
            break
        if same_section_only:
            prev_doc = prev_chunk.get("metadata", {}).get("file_name", "")
            prev_section = prev_chunk.get("metadata", {}).get("heading", "")
            if prev_doc != doc_name or prev_section != section:
                break
        collected.add(prev_id)
        prev_id = prev_chunk.get("prev_id")
        added += 1

    # Next direction
    next_id = chunk_info.get("next_id")
    added = 0
    while next_id and added < max_expansion:
        if next_id in collected:
            break
        next_chunk = all_chunks_map.get(next_id)
        if not next_chunk:
            break
        if same_section_only:
            next_doc = next_chunk.get("metadata", {}).get("file_name", "")
            next_section = next_chunk.get("metadata", {}).get("heading", "")
            if next_doc != doc_name or next_section != section:
                break
        collected.add(next_id)
        next_id = next_chunk.get("next_id")
        added += 1


def _build_consolidated_intervals(
    all_ids: set,
    all_chunks_map: Dict[str, Dict[str, Any]]
) -> List[List[str]]:
    """Gom IDs thành continuous ranges, merge overlapping intervals.

    ID numbering assumes sequential chunk ordering within a document.
    We group by (file_name, chunk_index range).
    """
    if not all_ids:
        return []

    # Group IDs by file_name
    by_file: Dict[str, List[Tuple[int, str]]] = {}
    for cid in all_ids:
        chunk_info = all_chunks_map.get(cid, {})
        meta = chunk_info.get("metadata", {})
        fname = meta.get("file_name", "_unknown_")
        idx = meta.get("chunk_index", -1)
        by_file.setdefault(fname, []).append((idx, cid))

    # Sort by chunk_index within each file, build intervals
    intervals = []
    for fname, items in by_file.items():
        items.sort(key=lambda x: x[0])
        current_interval = []
        prev_idx = -10  # sentinel
        for idx, cid in items:
            if prev_idx < 0 or idx <= prev_idx + 2:  # allow gap of 1
                current_interval.append(cid)
            else:
                if current_interval:
                    intervals.append(current_interval)
                current_interval = [cid]
            prev_idx = idx
        if current_interval:
            intervals.append(current_interval)

    return intervals


def _find_interval(chunk_id: str, intervals: List[List[str]]) -> List[str]:
    """Find the interval containing chunk_id."""
    for interval in intervals:
        if chunk_id in interval:
            return interval
    return []
