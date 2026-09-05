"""HybridSearcher — SRP: thực thi tra cứu kết hợp Vector Search + BM25 Search + RRF Fusion."""

from typing import List, Dict, Any, Tuple
from core.vector_store import ChromaVectorStore
from core.bm25_index import BM25Index
from retrieval.fusion import reciprocal_rank_fusion
from config import config


class HybridSearcher:
    """Điều phối tra cứu Hybrid (Semantic + BM25) và áp dụng RRF Fusion."""

    def __init__(self, vector_store: ChromaVectorStore, bm25_index: BM25Index = None):
        self.vector_store = vector_store
        self.bm25_index = bm25_index or BM25Index()

    def search(
        self,
        user_query: str,
        query_vector: List[float],
        semantic_top_k: int,
        bm25_top_k: int,
        final_top_k: int,
        intent: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Thực thi Semantic Search + BM25 Search -> RRF Fusion -> Intent Boosting."""
        # 1. Semantic search
        semantic_results_raw = self.vector_store.search_similarity(
            query_vector, top_k=semantic_top_k
        )
        semantic_tuples: List[Tuple[str, float, Dict[str, Any]]] = []
        for r in semantic_results_raw:
            meta = r.get("metadata", {})
            chunk_id = meta.get("chunk_id", "")
            if chunk_id:
                meta_with_text = {**meta, "text": r.get("text", "")}
                semantic_tuples.append((chunk_id, r.get("distance", 0), meta_with_text))

        # 2. BM25 search
        bm25_tuples: List[Tuple[str, float, Dict[str, Any]]] = []
        if config.BM25_ENABLED and self.bm25_index.is_loaded:
            bm25_tuples = self.bm25_index.search(user_query, top_k=bm25_top_k)

        # 3. RRF Fusion
        fused = reciprocal_rank_fusion(
            semantic_tuples, bm25_tuples,
            k=config.RRF_K, final_top_k=final_top_k
        )

        # 4. Intent-based boosting
        target_chapter = intent.get("chapter_match")
        boost_score = config.INTENT_BOOST_SCORE if hasattr(config, "INTENT_BOOST_SCORE") else 0.01
        if target_chapter:
            for r in fused:
                meta = r.get("metadata", {})
                chap = str(meta.get("chapter", ""))
                text = r.get("text", "")
                if (target_chapter in chap or
                    f"Chương {target_chapter}" in text or
                    f"CHƯƠNG {target_chapter}" in text or
                    target_chapter in str(meta.get("heading", ""))):
                    r["rrf_score"] = r.get("rrf_score", 0) + boost_score

        return fused
