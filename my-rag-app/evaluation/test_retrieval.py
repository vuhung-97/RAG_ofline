"""Test retrieval — Kiểm tra embedding + BM25 + RRF lấy chunks có đúng không.
Chỉ test retrieval, KHÔNG gọi LLM.

Cách chạy:
  python evaluation/test_retrieval.py
  python evaluation/test_retrieval.py "câu hỏi cần test"
"""

import sys
import os
import io
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from core.embedding_service import OllamaEmbeddingService
from core.vector_store import ChromaVectorStore
from core.bm25_index import BM25Index
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.dedup import deduplicate_chunks


def test_retrieval(query: str, top_k: int = 6):
    """Chạy retrieval pipeline và in kết quả raw chunks."""
    print(f"\nQuery: \"{query}\"")
    print(f"Embedding model: {config.EMBED_MODEL}")
    print(f"Semantic top_k: {top_k} | BM25 top_k: {top_k} | Final top_k: {top_k}")
    print("=" * 60)

    # 1. Init services
    t0 = time.time()
    embedding_service = OllamaEmbeddingService()
    vector_store = ChromaVectorStore()
    bm25_index = BM25Index()

    # 2. Embed query
    t_embed = time.time()
    query_vector = embedding_service.embed_query(query)
    t_embed = time.time() - t_embed
    print(f"\n[Embedding] {t_embed*1000:.0f}ms | dim={len(query_vector)}")

    # 3. Semantic search
    t_sem = time.time()
    semantic_raw = vector_store.search_similarity(query_vector, top_k=top_k)
    t_sem = time.time() - t_sem

    semantic_tuples = []
    for r in semantic_raw:
        meta = r.get("metadata", {})
        chunk_id = meta.get("chunk_id", "")
        if chunk_id:
            meta_with_text = {**meta, "text": r.get("text", "")}
            semantic_tuples.append((chunk_id, r.get("distance", 0), meta_with_text))

    print(f"[Semantic] {t_sem*1000:.0f}ms | {len(semantic_tuples)} results")
    for i, (cid, dist, meta) in enumerate(semantic_tuples):
        print(f"  #{i+1} dist={dist:.4f} | {meta.get('chapter','')} > {meta.get('heading','')[:50]}")

    # 4. BM25 search
    bm25_tuples = []
    if config.BM25_ENABLED:
        # Try load from disk first
        bm25_loaded = bm25_index.load()
        if not bm25_loaded:
            # Rebuild from ChromaDB
            print("[BM25]    Rebuilding from ChromaDB...")
            all_data = vector_store.collection.get()
            if all_data and all_data["ids"]:
                bm25_index.build(all_data["ids"], all_data["documents"], all_data["metadatas"])
                bm25_loaded = bm25_index.is_loaded
        if bm25_loaded:
            t_bm25 = time.time()
            bm25_tuples = bm25_index.search(query, top_k=top_k)
            t_bm25 = time.time() - t_bm25
            print(f"[BM25]    {t_bm25*1000:.0f}ms | {len(bm25_tuples)} results")
            for i, (cid, score, meta) in enumerate(bm25_tuples):
                print(f"  #{i+1} score={score:.4f} | {meta.get('chapter','')} > {meta.get('heading','')[:50]}")
        else:
            print("[BM25]    Failed to load/rebuild")
    else:
        print("[BM25]    Disabled")

    # 5. RRF Fusion
    fused = reciprocal_rank_fusion(
        semantic_tuples, bm25_tuples,
        k=config.RRF_K, final_top_k=top_k
    )

    # 6. Dedup
    deduped = deduplicate_chunks(fused)

    t_total = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"KẾT QUẢ CUỐI CÙNG (RRF + Dedup): {len(deduped)} chunks | {t_total*1000:.0f}ms")
    print(f"{'=' * 60}")

    for i, r in enumerate(deduped):
        meta = r.get("metadata", {})
        text = r.get("text", "")
        print(f"\n[{i+1}] rrf={r.get('rrf_score',0):.4f} | sem={r.get('semantic_score',0):.4f} | bm25={r.get('bm25_score',0):.4f}")
        print(f"    Chapter: {meta.get('chapter','?')} | Heading: {meta.get('heading','?')}")
        print(f"    Type: {meta.get('type','?')} | File: {meta.get('file_name','?')}")
        print(f"    Text: {text}")

    print(f"\n{'=' * 60}")
    print("CHUNK ANALYSIS:")
    print(f"{'=' * 60}")
    for i, r in enumerate(deduped):
        text_raw = r.get("text", "")
        word_count = len(text_raw.split())
        char_count = len(text_raw)
        est_tokens = char_count // 4
        line_count = len(text_raw.split("\n"))
        print(f"\n[Chunk {i+1}] {word_count} từ | {char_count} ký tự | ~{est_tokens} tokens | {line_count} dòng | index={r.get('metadata',{}).get('chunk_index','?')}")
        print(f"  Full:\n{text_raw}")

    return deduped


if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        test_retrieval(query)
    else:
        print("=== RETRIEVAL TEST (không gọi LLM) ===")
        print("Nhập câu hỏi để test retrieval (nhập 'q' để thoát):\n")

        while True:
            query = input("Query> ").strip()
            if not query or query.lower() == 'q':
                break
            test_retrieval(query)
            print()
