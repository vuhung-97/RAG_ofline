"""RRF (Reciprocal Rank Fusion) — SRP: chỉ fusion kết quả search."""

from typing import List, Dict, Any, Tuple


def reciprocal_rank_fusion(
    semantic_results: List[Tuple[str, float, Dict]],
    bm25_results: List[Tuple[str, float, Dict]],
    k: int = 60,
    final_top_k: int = 8
) -> List[Dict[str, Any]]:
    """
    Fusion semantic + BM25 results using Reciprocal Rank Fusion.

    Args:
        semantic_results: [(chunk_id, score, metadata), ...] sorted by score desc
        bm25_results: [(chunk_id, score, metadata), ...] sorted by score desc
        k: RRF constant (higher = less weight to top ranks)
        final_top_k: final number of results

    Returns:
        List of fused results sorted by RRF score descending
    """
    rrf_scores: Dict[str, float] = {}
    result_map: Dict[str, Dict[str, Any]] = {}

    # Process semantic results
    for rank, (chunk_id, score, metadata) in enumerate(semantic_results, 1):
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1.0 / (k + rank)
        if chunk_id not in result_map:
            result_map[chunk_id] = {
                "chunk_id": chunk_id,
                "text": metadata.get("text", ""),
                "metadata": metadata,
                "semantic_score": score,
                "bm25_score": 0.0,
                "rrf_score": 0.0
            }
        result_map[chunk_id]["semantic_score"] = score

    # Process BM25 results
    for rank, (chunk_id, score, metadata) in enumerate(bm25_results, 1):
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1.0 / (k + rank)
        if chunk_id not in result_map:
            result_map[chunk_id] = {
                "chunk_id": chunk_id,
                "text": metadata.get("text", ""),
                "metadata": metadata,
                "semantic_score": 0.0,
                "bm25_score": score,
                "rrf_score": 0.0
            }
        result_map[chunk_id]["bm25_score"] = score

    # Assign RRF scores
    for chunk_id, score in rrf_scores.items():
        result_map[chunk_id]["rrf_score"] = score

    # Sort by RRF score descending
    fused = sorted(result_map.values(), key=lambda x: -x["rrf_score"])
    return fused[:final_top_k]
