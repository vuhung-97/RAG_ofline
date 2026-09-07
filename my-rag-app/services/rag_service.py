"""RAGService — SRP: Điều phối luồng RAG Pipeline (Tra cứu -> Đóng gói Context -> LLM Stream)."""

import time
import logging
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
from retrieval.query_rewriter import FollowUpDetector, QueryRewriter
from core.reranker import LLMReranker
from config import config

logger = logging.getLogger("rag_pipeline")


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
        self.bm25_index = bm25_index
        self._update_hybrid_searcher()

    def _update_hybrid_searcher(self):
        """Cập nhật HybridSearcher với BM25 index hiện tại."""
        self.hybrid_searcher = HybridSearcher(self.vector_store, self.bm25_index)

    def set_bm25_index(self, bm25_index: BM25Index):
        """Đổi BM25 index (khi switch workspace)."""
        self.bm25_index = bm25_index
        self._update_hybrid_searcher()

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

    def _expand_tables(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fetch full table cho bất kỳ table chunk nào trong results."""
        table_chunks = [c for c in chunks if c.get("metadata", {}).get("table_id") is not None]
        non_table = [c for c in chunks if c.get("metadata", {}).get("table_id") is None]

        if not table_chunks or not config.TABLE_EXPANSION_ENABLED:
            return chunks

        # Group table chunks by (file_name, table_id)
        table_keys = set()
        for c in table_chunks:
            meta = c.get("metadata", {})
            key = (meta.get("file_name", ""), meta.get("table_id"))
            table_keys.add(key)

        # Fetch full tables from ChromaDB
        expanded_tables = []
        for file_name, table_id in table_keys:
            table_full = self._fetch_full_table(file_name, table_id)
            if table_full:
                expanded_tables.extend(table_full)

        # Dedup: chỉ giữ tables KHÔNG có trong non_table
        non_table_ids = {c.get("chunk_id") for c in non_table}
        new_tables = [t for t in expanded_tables if t.get("chunk_id") not in non_table_ids]

        if config.ENABLE_PIPELINE_DEBUG and new_tables:
            logger.debug(f"[TABLE EXPANSION] Fetched {len(new_tables)} table chunks for {len(table_keys)} tables")

        return non_table + new_tables

    def _fetch_full_table(self, file_name: str, table_id: int) -> List[Dict[str, Any]]:
        """Fetch all chunks of a specific table from ChromaDB."""
        try:
            all_data = self.vector_store.collection.get(
                where={"$and": [{"file_name": file_name}, {"table_id": table_id}]}
            )
            if not all_data or not all_data["ids"]:
                return []
            chunks = []
            for cid, doc, meta in zip(all_data["ids"], all_data["documents"], all_data["metadatas"]):
                chunks.append({
                    "chunk_id": cid,
                    "text": doc,
                    "metadata": meta,
                    "rrf_score": 0,
                })
            # Sort by chunk_index
            chunks.sort(key=lambda c: c["metadata"].get("chunk_index", 0))
            return chunks[:config.TABLE_MAX_CHUNKS]
        except Exception:
            return []

    def _validate_answer(self, answer: str, context: str, num_chunks: int = 0) -> str:
        """Ủy quyền kiểm tra ảo giác cho GuardrailValidator."""
        return GuardrailValidator.validate_answer(answer, context, num_chunks)

    def query(
        self,
        user_query: str,
        chat_history: List[Dict[str, str]] = None,
        llm_model: str = config.LLM_MODEL,
        embed_model: str = config.EMBED_MODEL,
        top_k: int = config.FINAL_TOP_K,
        num_ctx: int = config.LLM_NUM_CTX,
        temperature: float = config.TEMPERATURE,
        enable_thinking: bool = config.ENABLE_THINKING,
        enable_rerank: bool = config.ENABLE_RERANK
    ) -> Dict[str, Any]:
        """Xử lý câu hỏi: Hybrid search -> Dedup -> Rerank -> Neighbor -> Table -> Context -> LLM stream."""
        t_pipeline_start = time.perf_counter()
        debug = config.ENABLE_PIPELINE_DEBUG

        if debug:
            logger.debug(f"[STEP 0] Query: \"{user_query}\"")

        # 0. Follow-up detection + query rewrite
        if chat_history and FollowUpDetector.is_follow_up(user_query, chat_history):
            rewritten = QueryRewriter.rewrite(user_query, chat_history)
            if rewritten != user_query:
                print(f"[RAG STEP 0] 🔄 Query rewrite: \"{rewritten}\"")
                if debug:
                    logger.debug(f"[STEP 0] Rewrite: \"{user_query}\" → \"{rewritten}\"")
                user_query = rewritten

        # 1. Intent detection (cho MỌI query, không chỉ summary)
        intent = IntentAnalyzer.detect_intent(user_query)
        is_summary = intent.get("is_summary", False)

        if debug:
            logger.debug(f"[STEP 0] Intent: summary={is_summary}, chapter={intent.get('chapter_match')}, "
                        f"section={intent.get('section_match')}, article={intent.get('article_match')}")

        # 2. Embed query
        t0 = time.perf_counter()
        query_vector = self.embedding_service.embed_query(user_query, model_name=embed_model)
        t_embed = (time.perf_counter() - t0) * 1000
        print(f"[RAG STEP 1] 🧠 Tạo Query Embedding ({embed_model}) ... [Xong: {t_embed:.1f} ms]")
        if debug:
            logger.debug(f"[STEP 1] Embed: {t_embed:.1f}ms")

        # 3. Hybrid search (Semantic + BM25 + RRF)
        t0 = time.perf_counter()
        fused_results = self.hybrid_searcher.search(
            user_query, query_vector,
            semantic_top_k=config.SEMANTIC_TOP_K if not is_summary else config.SEMANTIC_TOP_K * 2,
            bm25_top_k=config.BM25_TOP_K if not is_summary else config.BM25_TOP_K * 2,
            fusion_top_k=config.FUSION_TOP_K if not is_summary else config.FUSION_TOP_K * 2,
            intent=intent
        )
        t_search = (time.perf_counter() - t0) * 1000
        print(f"[RAG STEP 2] 🔍 Hybrid Search (Semantic + BM25) & RRF ... [Xong: {t_search:.1f} ms | Tìm thấy: {len(fused_results)} chunks]")
        if debug:
            logger.debug(f"[STEP 2] Fused: {len(fused_results)} chunks ({t_search:.1f}ms)")

        # 4. Dedup
        deduped = deduplicate_chunks(fused_results)
        if debug:
            logger.debug(f"[STEP 3] Dedup: {len(fused_results)} → {len(deduped)} chunks")

        # 5. Rerank (trước neighbor expansion để đánh giá chunk gốc, skip summary)
        t0 = time.perf_counter()
        do_rerank = enable_rerank and not is_summary
        if do_rerank and len(deduped) > top_k:
            reranker = LLMReranker()
            deduped = reranker.rerank(user_query, deduped, llm_model, top_k=top_k)
        t_rerank = (time.perf_counter() - t0) * 1000
        if do_rerank:
            print(f"[RAG STEP 3] 🏆 Rerank ... [Xong: {t_rerank:.1f} ms]")
        if debug:
            logger.debug(f"[STEP 3] Rerank: {len(deduped)} chunks ({t_rerank:.1f}ms)")

        # 6. Neighbor expansion
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
        print(f"[RAG STEP 4] 🔗 Mở rộng đoạn lân cận (Neighbor Expansion) ... [Xong: {t_neighbor:.1f} ms]")

        # 7. Table expansion
        t0 = time.perf_counter()
        if config.TABLE_EXPANSION_ENABLED and expanded:
            expanded = self._expand_tables(expanded)
        t_table = (time.perf_counter() - t0) * 1000
        if debug:
            logger.debug(f"[STEP 4] Table expansion: {t_table:.1f}ms, total chunks: {len(expanded)}")

        # 8. Context budget & formatting
        final_chunks = expanded[:config.FINAL_TOP_K]
        t0 = time.perf_counter()
        formatted_context, merged_chunks = build_context(
            final_chunks,
            max_chunks=config.FINAL_TOP_K,
            max_tokens=config.CONTEXT_BUDGET_EVIDENCE
        )
        t_context = (time.perf_counter() - t0) * 1000

        num_chunks = len(merged_chunks)
        print(f"[RAG STEP 5] 📄 Đóng gói Ngữ cảnh (Context Builder) ... [Xong: {t_context:.1f} ms | Đã dùng: {num_chunks} chunks]")
        if debug:
            logger.debug(f"[STEP 5] Context: {num_chunks} chunks, ~{len(formatted_context)//4} tokens")

        sources = format_sources_for_ui(merged_chunks)

        # 9. Relevance check
        if not formatted_context or not formatted_context.strip() or not has_sufficient_relevance(merged_chunks):
            no_result_msg = self._get_no_result_message(intent)
            print("[RAG PIPELINE] ⚠️ Không có ngữ cảnh đủ độ liên quan để trả lời.")
            if debug:
                logger.debug("[STEP 5] Relevance: FAIL — no sufficient context")
            return {
                "stream": iter([no_result_msg]),
                "sources": sources,
                "no_context": True
            }

        if debug:
            logger.debug(f"[STEP 5] Relevance: PASS")

        # 10. Build Prompt & Messages via PromptBuilder
        system_content = PromptBuilder.build_system_content(is_summary, formatted_context, num_chunks)
        messages = PromptBuilder.build_messages(system_content, user_query, chat_history, enable_thinking)
        if debug:
            logger.debug(f"[STEP 6] Prompt: ~{sum(len(m.get('content',''))//4 for m in messages)} tokens")

        # 11. Stream LLM
        print(f"[LLM STEP 6] 🤖 Đã gửi Prompt sang LLM ({llm_model}) ... Đang chờ phản hồi...")
        stream_generator = self.llm_service.stream_chat(
            messages=messages,
            model_name=llm_model,
            num_ctx=num_ctx,
            temperature=temperature,
            enable_thinking=enable_thinking
        )

        t_total = (time.perf_counter() - t_pipeline_start) * 1000
        if debug:
            logger.debug(f"[PIPELINE TOTAL] {t_total:.1f}ms")

        return {
            "stream": stream_generator,
            "sources": sources,
            "merged_chunks": merged_chunks
        }

    @staticmethod
    def _get_no_result_message(intent: Dict[str, Any]) -> str:
        """Trả về message phù hợp khi không tìm thấy kết quả."""
        return "Không tìm thấy thông tin phù hợp trong tài liệu được cung cấp."
