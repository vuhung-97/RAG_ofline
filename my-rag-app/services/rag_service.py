"""RAGService — SRP: Điều phối luồng RAG Pipeline (Tra cứu -> Đóng gói Context -> LLM Stream)."""

import time
from typing import List, Dict, Any, Generator

from services.embedding_service import OllamaEmbeddingService
from services.llm_service import OllamaLLMService
from core.vector_store import ChromaVectorStore
from core.bm25_index import BM25Index

from retrieval.intent_analyzer import IntentAnalyzer
from retrieval.prompt_builder import PromptBuilder
from retrieval.guardrail import GuardrailValidator
from retrieval.hybrid_searcher import HybridSearcher
from retrieval.dedup import deduplicate_chunks
from retrieval.neighbor_expander import expand_neighbors
from retrieval.context_builder import build_context, format_sources_for_ui
from retrieval.relevance_checker import has_sufficient_relevance
from config import config


class RAGService:
    """SRP & Orchestration: Điều phối luồng tra cứu RAG (Hybrid Search -> Context -> LLM Stream)."""

    def __init__(
        self,
        embedding_service: OllamaEmbeddingService,
        vector_store: ChromaVectorStore,
        llm_service: OllamaLLMService,
        bm25_index: BM25Index = None
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.llm_service = llm_service
        self.hybrid_searcher = HybridSearcher(vector_store, bm25_index)

    def _get_neighbor_chunks_map(self, chunks: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Lấy thông tin của các chunks hiện tại và prev/next IDs từ ChromaDB."""
        chunks_map = {}
        if not chunks:
            return chunks_map

        needed_ids = set()
        for c in chunks:
            meta = c.get("metadata", {})
            cid = c.get("chunk_id") or meta.get("chunk_id", "")
            if cid:
                needed_ids.add(cid)
            prev_id = meta.get("previous_chunk_id")
            next_id = meta.get("next_chunk_id")
            if prev_id:
                needed_ids.add(prev_id)
            if next_id:
                needed_ids.add(next_id)

        if not needed_ids:
            return chunks_map

        try:
            matched_data = self.vector_store.collection.get(ids=list(needed_ids))
            if matched_data and matched_data["ids"]:
                for cid, doc, meta in zip(matched_data["ids"], matched_data["documents"], matched_data["metadatas"]):
                    chunks_map[cid] = {
                        "text": doc,
                        "metadata": meta,
                        "prev_id": meta.get("previous_chunk_id"),
                        "next_id": meta.get("next_chunk_id"),
                    }
        except Exception:
            pass
        return chunks_map

    def _validate_answer(self, answer: str, context: str, num_chunks: int = 0) -> str:
        """Ủy quyền kiểm tra ảo giác cho GuardrailValidator."""
        return GuardrailValidator.validate_answer(answer, context, num_chunks)

    def query(
        self,
        user_query: str,
        chat_history: List[Dict[str, str]] = None,
        llm_model: str = config.LLM_MODEL,
        embed_model: str = config.EMBED_MODEL,
        top_k: int = config.TOP_K,
        num_ctx: int = config.LLM_NUM_CTX,
        temperature: float = config.TEMPERATURE,
        enable_thinking: bool = config.ENABLE_THINKING,
        enable_rerank: bool = config.ENABLE_RERANK
    ) -> Dict[str, Any]:
        """Xử lý câu hỏi: Hybrid search → Dedup → Neighbor → Context → LLM stream."""
        print(f"\n[RAG PIPELINE] 🚀 Bắt đầu truy vấn: \"{user_query}\" (LLM: {llm_model} | Embed: {embed_model})")

        # 1. Intent detection
        intent = IntentAnalyzer.detect_intent(user_query)
        is_summary = intent.get("is_summary", False)

        search_top_k = top_k * 5 if is_summary else top_k
        summary_num_ctx = max(num_ctx, 8192) if is_summary else num_ctx

        # 2. Embed query
        t0 = time.perf_counter()
        query_vector = self.embedding_service.embed_query(user_query, model_name=embed_model)
        t_embed = (time.perf_counter() - t0) * 1000
        print(f"[RAG STEP 1] 🧠 Tạo Query Embedding ({embed_model}) ... [Xong: {t_embed:.1f} ms]")

        # 3. Hybrid search (Semantic + BM25 + RRF)
        t0 = time.perf_counter()
        fused_results = self.hybrid_searcher.search(
            user_query, query_vector,
            semantic_top_k=search_top_k,
            bm25_top_k=search_top_k,
            final_top_k=search_top_k,
            intent=intent
        )
        t_search = (time.perf_counter() - t0) * 1000
        print(f"[RAG STEP 2] 🔍 Hybrid Search (Semantic + BM25) & RRF ... [Xong: {t_search:.1f} ms | Tìm thấy: {len(fused_results)} chunks]")

        # 4. Dedup
        deduped = deduplicate_chunks(fused_results)

        # 5. Neighbor expansion
        t0 = time.perf_counter()
        if config.ENABLE_NEIGHBOR_EXPANSION and deduped:
            all_chunks_map = self._get_neighbor_chunks_map(deduped)
            expanded = expand_neighbors(
                deduped, all_chunks_map,
                max_expansion=config.NEIGHBOR_MAX_EXPANSION,
                same_section_only=config.NEIGHBOR_SAME_SECTION_ONLY
            )
        else:
            expanded = deduped
        t_neighbor = (time.perf_counter() - t0) * 1000
        print(f"[RAG STEP 3] 🔗 Mở rộng đoạn lân cận (Neighbor Expansion) ... [Xong: {t_neighbor:.1f} ms]")

        # 6. Context budget & formatting
        final_chunks = expanded[:search_top_k]
        t0 = time.perf_counter()
        dynamic_max_tokens = max(1000, num_ctx - 1000)
        summary_max_tokens = summary_num_ctx - 1000 if is_summary else dynamic_max_tokens
        formatted_context, merged_chunks = build_context(
            final_chunks,
            max_chunks=search_top_k,
            max_tokens=summary_max_tokens
        )
        t_context = (time.perf_counter() - t0) * 1000

        num_chunks = len(merged_chunks)
        print(f"[RAG STEP 4] 📄 Đóng gói Ngữ cảnh (Context Builder) ... [Xong: {t_context:.1f} ms | Đã dùng: {num_chunks} chunks]")

        sources = format_sources_for_ui(merged_chunks)

        # 7. Relevance check
        if not formatted_context or not formatted_context.strip() or not has_sufficient_relevance(merged_chunks):
            no_result_msg = self._get_no_result_message(intent)
            print("[RAG PIPELINE] ⚠️ Không có ngữ cảnh đủ độ liên quan để trả lời.")
            return {
                "stream": iter([no_result_msg]),
                "sources": sources,
                "no_context": True
            }

        # 8. Build Prompt & Messages via PromptBuilder
        system_content = PromptBuilder.build_system_content(is_summary, formatted_context, num_chunks)
        messages = PromptBuilder.build_messages(system_content, user_query, chat_history, enable_thinking)

        # 9. Stream LLM
        print(f"[LLM STEP 5] 🤖 Đã gửi Prompt sang LLM ({llm_model}) ... Đang chờ phản hồi...")
        stream_generator = self.llm_service.stream_chat(
            messages=messages,
            model_name=llm_model,
            num_ctx=summary_num_ctx,
            temperature=temperature,
            enable_thinking=enable_thinking
        )

        return {
            "stream": stream_generator,
            "sources": sources,
            "merged_chunks": merged_chunks
        }

    @staticmethod
    def _get_no_result_message(intent: Dict[str, Any]) -> str:
        """Trả về message phù hợp khi không tìm thấy kết quả."""
        return "Không tìm thấy thông tin phù hợp trong tài liệu được cung cấp."
