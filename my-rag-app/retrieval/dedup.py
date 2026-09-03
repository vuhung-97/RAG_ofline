"""Deduplication — SRP: chỉ loại bỏ chunk trùng lặp hoặc overlap cao."""

from typing import List, Dict, Any


def deduplicate_chunks(
    results: List[Dict[str, Any]],
    text_threshold: float = 0.85
) -> List[Dict[str, Any]]:
    """
    Loại bỏ chunk trùng nội dung hoặc overlap quá cao.
    Uses simple Jaccard text similarity — no model needed.
    """
    if not results:
        return results

    seen_hashes: List[str] = []
    deduped: List[Dict[str, Any]] = []

    for result in results:
        text = result.get("text", "").strip()
        if not text:
            continue

        is_duplicate = False
        text_lower = text.lower()
        for seen in seen_hashes:
            similarity = _jaccard_similarity(text_lower, seen)
            if similarity >= text_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            deduped.append(result)
            seen_hashes.append(text_lower)

    return deduped


def _jaccard_similarity(a: str, b: str) -> float:
    """Jaccard similarity cho 2 strings."""
    set_a = set(a.split())
    set_b = set(b.split())
    if not set_a or not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)
