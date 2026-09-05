"""Benchmark script — Đánh giá retrieval + generation quality."""

import sys
import os
import io
import json
import time
from typing import Dict, Any, List

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from core.markitdown_loader import MarkItDownLoader
from services.embedding_service import OllamaEmbeddingService
from core.vector_store import ChromaVectorStore
from services.llm_service import OllamaLLMService
from core.bm25_index import BM25Index
from services.document_service import DocumentService
from services.rag_service import RAGService
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.dedup import deduplicate_chunks
from retrieval.context_builder import build_context


def load_benchmark(path: str) -> Dict:
    """Load benchmark dataset."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_document_indexed(doc_service: DocumentService, file_path: str, file_name: str):
    """Ensure document is indexed before benchmark."""
    indexed = doc_service.vector_store.get_indexed_files()
    if file_name not in indexed:
        print(f"Indexing document: {file_name}...")
        result = doc_service.process_and_index_file(file_path, file_name)
        print(f"  Result: {result.get('message', 'Unknown')}")
    else:
        print(f"Document already indexed: {file_name}")


def evaluate_retrieval(
    rag_service: RAGService,
    query: str,
    expected_section: str
) -> Dict[str, Any]:
    """Evaluate retrieval quality for a single query."""
    # Embed query - use embed_text (no prefix) for compatibility with existing index
    # To use prefix, re-index all documents first
    query_vector = rag_service.embedding_service.embed_text(query)

    # Hybrid search
    intent = rag_service._detect_intent(query)
    fused = rag_service._hybrid_search(
        query, query_vector,
        semantic_top_k=config.SEMANTIC_TOP_K,
        bm25_top_k=config.BM25_TOP_K,
        final_top_k=config.FINAL_TOP_K,
        intent=intent
    )

    # Dedup
    deduped = deduplicate_chunks(fused)

    # Check if expected section is in results
    found_section = False
    found_keywords = []
    for r in deduped:
        meta = r.get("metadata", {})
        heading = meta.get("heading", "")
        chapter = meta.get("chapter", "")
        if expected_section and (expected_section in heading or expected_section in chapter):
            found_section = True

    # Get retrieved chunk IDs
    retrieved_ids = [r.get("chunk_id", "") for r in deduped]
    retrieved_sections = []
    for r in deduped:
        meta = r.get("metadata", {})
        section = meta.get("heading", "")
        if section and section not in retrieved_sections:
            retrieved_sections.append(section)

    return {
        "found_section": found_section,
        "retrieved_count": len(deduped),
        "retrieved_ids": retrieved_ids,
        "retrieved_sections": retrieved_sections[:5],
        "rrf_scores": [round(r.get("rrf_score", 0), 4) for r in deduped[:3]],
    }


def run_benchmark(benchmark_path: str, document_path: str = None):
    """Run full benchmark."""
    dataset = load_benchmark(benchmark_path)
    doc_name = dataset["document"]

    if document_path is None:
        document_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "temp", doc_name
        )

    # Initialize services
    print("Initializing services...")
    md_loader = MarkItDownLoader()
    embedding_service = OllamaEmbeddingService()
    vector_store = ChromaVectorStore()
    llm_service = OllamaLLMService()
    bm25_index = BM25Index()

    # Load existing BM25 index
    if config.BM25_ENABLED:
        bm25_index.load()

    doc_service = DocumentService(
        md_loader=md_loader,
        embedding_service=embedding_service,
        vector_store=vector_store,
        bm25_index=bm25_index
    )

    rag_service = RAGService(
        embedding_service=embedding_service,
        vector_store=vector_store,
        llm_service=llm_service,
        bm25_index=bm25_index
    )

    # Ensure document is indexed
    if os.path.exists(document_path):
        ensure_document_indexed(doc_service, document_path, doc_name)
        # Rebuild BM25 from ChromaDB data
        if config.BM25_ENABLED:
            doc_service.rebuild_bm25_index()
            print(f"BM25 index: {bm25_index.size} documents")
    else:
        print(f"Document not found: {document_path}")
        print("Running benchmark with existing index only...")

    # Run benchmark
    questions = dataset["questions"]
    results = []
    recall_counts = {1: 0, 3: 0, 5: 0, 10: 0}
    total_answerable = sum(1 for q in questions if q["answerable"])

    print(f"\nRunning benchmark: {len(questions)} questions, {total_answerable} answerable")
    print("=" * 70)

    for i, q in enumerate(questions, 1):
        query = q["question"]
        expected_section = q["expected_sources"][0]["section"] if q["expected_sources"] else ""
        is_answerable = q["answerable"]

        print(f"\n[{i}/{len(questions)}] {q['category']}: {query[:80]}...")

        start_time = time.perf_counter()
        retrieval_result = evaluate_retrieval(rag_service, query, expected_section)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Recall@K - if section found at ANY rank, it counts for ALL K
        if is_answerable:
            found = retrieval_result["found_section"]
            if found:
                for k in [1, 3, 5, 10]:
                    recall_counts[k] += 1

        result = {
            "id": q["id"],
            "query": query[:100],
            "category": q["category"],
            "answerable": is_answerable,
            "retrieval": retrieval_result,
            "latency_ms": round(elapsed_ms, 1)
        }
        results.append(result)

        status = "OK" if (not is_answerable or retrieval_result["found_section"]) else "MISS"
        print(f"  {status} | chunks={retrieval_result['retrieved_count']} | "
              f"rrf={retrieval_result['rrf_scores'][:2]} | "
              f"sections={retrieval_result['retrieved_sections'][:3]} | "
              f"{elapsed_ms:.0f}ms")

    # Summary
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"Total questions: {len(questions)}")
    print(f"Answerable: {total_answerable}")
    print(f"Retrieved sections found:")
    for k in [1, 3, 5, 10]:
        pct = (recall_counts[k] / total_answerable * 100) if total_answerable > 0 else 0
        print(f"  Recall@{k}: {recall_counts[k]}/{total_answerable} ({pct:.1f}%)")

    avg_latency = sum(r["latency_ms"] for r in results) / len(results) if results else 0
    print(f"Average retrieval latency: {avg_latency:.1f}ms")

    # Save results
    output_path = benchmark_path.replace(".json", "_results.json")
    output = {
        "dataset_version": dataset["version"],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "semantic_top_k": config.SEMANTIC_TOP_K,
            "bm25_top_k": config.BM25_TOP_K,
            "final_top_k": config.FINAL_TOP_K,
            "rrf_k": config.RRF_K,
            "max_context_chunks": f"dynamic (top_k * 2)",
            "max_context_tokens": config.MAX_CONTEXT_TOKENS,
        },
        "summary": {
            "total": len(questions),
            "answerable": total_answerable,
            "recall@k": {str(k): f"{recall_counts[k]}/{total_answerable}" for k in [1, 3, 5, 10]},
            "avg_latency_ms": round(avg_latency, 1),
        },
        "results": results
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="RAG Benchmark")
    parser.add_argument("--dataset", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "datasets", "benchmark_v1.json"
    ))
    parser.add_argument("--document", default=None)
    args = parser.parse_args()

    run_benchmark(args.dataset, args.document)
