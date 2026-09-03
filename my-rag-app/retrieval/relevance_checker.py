"""Relevance checker — SRP: chỉ quyết định có đủ thông tin để trả lời."""

from typing import List, Dict, Any


def has_sufficient_relevance(
    fused_results: List[Dict[str, Any]],
    min_score: float = 0.015,
    min_results: int = 1,
    max_distance: float = 0.9
) -> bool:
    """
    Kiểm tra xem kết quả retrieval có đủ tốt để trả lời.

    Multi-signal check:
    - Có ít nhất min_results kết quả
    - Top-1 RRF score > min_score
    - Top-1 distance < max_distance (quality gate)
    """
    if not fused_results:
        return False

    if len(fused_results) < min_results:
        return False

    top1 = fused_results[0]
    top1_rrf = top1.get("rrf_score", 0.0)
    if top1_rrf < min_score:
        return False

    # Check semantic distance quality
    top1_distance = top1.get("distance", 0.0)
    if top1_distance > max_distance:
        return False

    return True


def compute_confidence(fused_results: List[Dict[str, Any]]) -> float:
    """
    Compute confidence score 0.0 - 1.0 dựa trên retrieval signals.
    Higher = more confident that we have relevant context.
    """
    if not fused_results:
        return 0.0

    top1 = fused_results[0]

    # Signal 1: RRF score (higher is better)
    rrf_score = top1.get("rrf_score", 0.0)
    # Normalize: typical RRF scores range 0.001 - 0.1
    rrf_signal = min(rrf_score * 10, 1.0)

    # Signal 2: Agreement (both semantic and BM25 found it)
    agreement_count = sum(
        1 for r in fused_results[:5]
        if r.get("semantic_score", 0) > 0 and r.get("bm25_score", 0) > 0
    )
    agreement_signal = min(agreement_count / 3, 1.0)

    # Signal 3: Number of results (more = more evidence)
    count_signal = min(len(fused_results) / 5, 1.0)

    # Signal 4: Distance quality (lower distance = better)
    distance = top1.get("distance", 1.0)
    distance_signal = max(0, 1.0 - distance)

    # Weighted combination
    confidence = (
        0.4 * rrf_signal +
        0.2 * agreement_signal +
        0.2 * count_signal +
        0.2 * distance_signal
    )

    return round(confidence, 3)
