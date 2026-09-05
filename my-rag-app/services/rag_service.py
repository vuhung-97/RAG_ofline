import time
from typing import List, Dict, Any, Generator
from core.embedding_service import OllamaEmbeddingService
from core.vector_store import ChromaVectorStore
from core.llm_service import OllamaLLMService
from core.bm25_index import BM25Index
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.dedup import deduplicate_chunks
from retrieval.neighbor_expander import expand_neighbors
from retrieval.context_builder import build_context, format_sources_for_ui
from retrieval.relevance_checker import has_sufficient_relevance
from config import config

import re as _re

SYSTEM_PROMPT = """Bạn là trợ lý tra cứu tài liệu. Trả lời câu hỏi dựa trên ngữ cảnh.

NGUYÊN TẮC TUYỆT ĐỐI:
- CHỈ dùng thông tin CÓ TRONG Context bên dưới.
- KHÔNG bịa đặt, KHÔNG thêm kiến thức ngoài Context.
- Nếu Context KHÔNG có thông tin → trả lời NGAY: "Tài liệu không đề cập đến thông tin này."
- Mỗi dòng phải có một nhãn nguồn [1], [2]; nhãn nguồn chỉ đặt ở cuối dòng; chỉ dùng nhãn từ [1] đến [{num_chunks}].
- Nếu Context thiếu thông tin về một bullet → KHÔNG bịa citation cho bullet đó.
- KHÔNG viết danh sách "Citations:", "Trích nguồn:" hay "Tài liệu tham khảo:" ở cuối bài.


CẤU TRÚC CÂU TRẢ LỜI:
- Mỗi ý chính xuống dòng mới.
- Cuối mỗi bullet có gán nhãn nguồn tham khảo [1], [2]
- Dùng bullet (-) hoặc numbered list (1. 2. 3.).


Ngữ cảnh:
{context}"""

SUMMARY_PROMPT = """Bạn là trợ lý tóm tắt tài liệu. Tóm tắt dựa trên ngữ cảnh.

NGUYÊN TẮC TUYỆT ĐỐI:
- CHỈ tóm tắt nội dung CÓ TRONG Context bên dưới.
- KHÔNG bịa đặt, KHÔNG thêm thông tin ngoài Context.
- Nếu Context KHÔNG có thông tin → ghi: "Phần này không có trong tài liệu."
- Nhãn nguồn chỉ đặt ở cuối bullet. Chỉ dùng nhãn từ [1] đến [{num_chunks}].
- KHÔNG tự tạo khối "Citations:", "Trích nguồn:" hoặc "Tài liệu tham khảo:" ở cuối câu trả lời.
- Nếu Context thiếu thông tin về một bullet → KHÔNG bịa citation cho bullet đó.


CẤU TRÚC CÂU TRẢ LỜI:
- Mỗi ý chính dùng bullet (-).
- Đặt nhãn trích dẫn [1], [2] trực tiếp ở cuối mỗi dòng bullet.
- Xuống dòng rõ ràng, dễ đọc.


Ngữ cảnh:
{context}"""

def _format_prompt(template: str, context: str, num_chunks: int) -> str:
    """Format prompt template."""
    return template.format(context=context, num_chunks=num_chunks)


# def build_references_section(answer_text: str, merged_chunks: List[Dict[str, Any]]) -> str:
#     """
#     Tạo phần 'Tài liệu tham khảo' tự động từ câu trả lời của LLM.
#     Đọc nhãn [1], [2]... trong answer_text → lấy raw text + metadata từ merged_chunks.
#     Đảm bảo 100% nguyên văn, không qua xử lý LLM.
#     """
#     cited_indices = sorted(list(set(int(m) for m in _re.findall(r'\[(\d+)\]', answer_text))))

#     valid_references = []
#     for idx in cited_indices:
#         if 1 <= idx <= len(merged_chunks):
#             chunk = merged_chunks[idx - 1]
#             meta = chunk.get("metadata", {})
#             raw_text = chunk.get("text", "").strip()

#             clean_snippet = _re.sub(r'\s+', ' ', raw_text)
#             if len(clean_snippet) > 180:
#                 clean_snippet = clean_snippet[:175] + "..."

#             meta_parts = []
#             if meta.get("file_name"):
#                 meta_parts.append(f"Tài liệu: {meta['file_name']}")
#             if meta.get("chapter"):
#                 meta_parts.append(f"Chương: {meta['chapter']}")
#             if meta.get("heading"):
#                 meta_parts.append(f"Mục: {meta['heading']}")
#             elif meta.get("page"):
#                 meta_parts.append(f"Trang: {meta['page']}")

#             meta_str = " | ".join(meta_parts)
#             ref_header = f"({meta_str})" if meta_str else ""

#             valid_references.append(f" [{idx}] {ref_header} \"{clean_snippet}\"")

#     if not valid_references:
#         return ""

#     return "\n\nTài liệu tham khảo:\n" + "\n".join(valid_references)


class RAGService:
    """SRP & Orchestration: Hybrid search pipeline + LLM generation."""

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
        self.bm25_index = bm25_index or BM25Index()

    def _validate_answer(self, answer: str, context: str, num_chunks: int = 0) -> str:
        """Guardrail: Kiểm tra hallucination bằng word overlap + length check."""
        # Defense-in-depth: loại bỏ <think>...</think> nếu còn sót
        answer = _re.sub(r"<think>.*?</think>", "", answer, flags=_re.DOTALL).strip()
        answer_lower = answer.lower().strip()

        # 1. Fallback phrases → cho qua
        fallback_phrases = [
            "tài liệu không đề cập",
            "không tìm thấy thông tin",
            "không có nội dung",
            "không tìm thấy vị trí",
            "phần này không có trong tài liệu",
            "không biết",
            "không chắc"
        ]
        if any(p in answer_lower for p in fallback_phrases):
            return answer

        # 2. Citation hợp lệ [1]-[num_chunks] → cho qua
        if num_chunks > 0:
            citations = _re.findall(r'\[(\d+)\]', answer)
            if citations:
                max_cite = max(int(c) for c in citations)
                if max_cite <= num_chunks:
                    return answer

        # 3. No context → reject
        if not context or context.strip() == "":
            return "Tài liệu không đề cập đến thông tin này."

        # 4. Length check: answer > 3x context → reject
        if len(answer) > len(context) * 3 and len(answer) > 500:
            return "Tài liệu không đề cập đến thông tin này."

        # 5. Word overlap check (chỉ cho answer dài)
        if len(answer) > 100:
            answer_words = set(_re.findall(r'\w{4,}', answer_lower))
            context_words = set(_re.findall(r'\w{4,}', context.lower()))
            if answer_words:
                overlap = len(answer_words & context_words) / len(answer_words)
                if overlap < 0.3:
                    return "Tài liệu không đề cập đến thông tin này."

        return answer

    def _detect_intent(self, user_query: str) -> Dict[str, Any]:
        """Phân tích intent của user query."""
        q_low = user_query.strip().lower()
        intent = {
            "is_summary": False,
            "chapter_match": None,
        }

        if q_low.startswith(("/tomtat", "/tonghop")):
            intent["is_summary"] = True
            m = _re.search(r"chương\s*(\d+)|chuong\s*(\d+)", user_query, _re.I)
            if m:
                intent["chapter_match"] = m.group(1) or m.group(2)
        elif _re.search(r"tóm tắt|tom tat|tổng hợp|tong hop", user_query, _re.I):
            m = _re.search(r"chương\s*(\d+)|chuong\s*(\d+)|mục\s*([\d\.]+)|muc\s*([\d\.]+)", user_query, _re.I)
            if m:
                intent["chapter_match"] = m.group(1) or m.group(2) or m.group(3) or m.group(4)
            intent["is_summary"] = True

        return intent

    def _get_neighbor_chunks_map(self, chunks: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Chỉ lấy thông tin của các chunks hiện tại và prev/next IDs của chúng từ ChromaDB thay vì query toàn bộ DB."""
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

    def _hybrid_search(
        self,
        user_query: str,
        query_vector: List[float],
        semantic_top_k: int,
        bm25_top_k: int,
        final_top_k: int,
        intent: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Hybrid search: Semantic + BM25 + RRF Fusion."""
        # Semantic search
        semantic_results_raw = self.vector_store.search_similarity(
            query_vector, top_k=semantic_top_k
        )
        # Convert to tuple format: (chunk_id, score, metadata)
        semantic_tuples = []
        for r in semantic_results_raw:
            meta = r.get("metadata", {})
            chunk_id = meta.get("chunk_id", "")
            if chunk_id:
                # Include text in metadata for later use
                meta_with_text = {**meta, "text": r.get("text", "")}
                semantic_tuples.append((chunk_id, r.get("distance", 0), meta_with_text))

        # BM25 search
        bm25_tuples = []
        if config.BM25_ENABLED and self.bm25_index.is_loaded:
            bm25_raw = self.bm25_index.search(user_query, top_k=bm25_top_k)
            bm25_tuples = bm25_raw

        # RRF Fusion
        fused = reciprocal_rank_fusion(
            semantic_tuples, bm25_tuples,
            k=config.RRF_K, final_top_k=final_top_k
        )

        # Apply intent-based boosting
        target_chapter = intent.get("chapter_match")
        if target_chapter:
            for r in fused:
                meta = r.get("metadata", {})
                chap = str(meta.get("chapter", ""))
                text = r.get("text", "")
                if (target_chapter in chap or
                    f"Chương {target_chapter}" in text or
                    f"CHƯƠNG {target_chapter}" in text or
                    target_chapter in str(meta.get("heading", ""))):
                    r["rrf_score"] = r.get("rrf_score", 0) + 0.01

        return fused

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
        t_total_start = time.perf_counter()
        print(f"\n[RAG PIPELINE] 🚀 Bắt đầu truy vấn: \"{user_query}\" (LLM: {llm_model} | Embed: {embed_model})")

        # 1. Intent detection
        intent = self._detect_intent(user_query)
        is_summary = intent.get("is_summary", False)

        # Summary: lấy nhiều chunks hơn, num_ctx lớn hơn
        search_top_k = top_k * 5 if is_summary else top_k
        summary_num_ctx = max(num_ctx, 8192) if is_summary else num_ctx

        # 2. Embed query với prefix
        t0 = time.perf_counter()
        query_vector = self.embedding_service.embed_query(user_query, model_name=embed_model)
        t_embed = (time.perf_counter() - t0) * 1000
        print(f"[RAG STEP 1] 🧠 Tạo Query Embedding ({embed_model}) ... [Xong: {t_embed:.1f} ms]")

        # 3. Hybrid search (Semantic + BM25 + RRF)
        t0 = time.perf_counter()
        fused_results = self._hybrid_search(
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

        # 6. Context budget
        final_chunks = expanded[:search_top_k]

        # 7. Format context
        t0 = time.perf_counter()
        dynamic_max_tokens = max(1000, num_ctx - 1000)
        summary_max_tokens = summary_num_ctx - 1000 if is_summary else dynamic_max_tokens
        formatted_context, merged_chunks = build_context(
            final_chunks,
            max_chunks=search_top_k,
            max_tokens=summary_max_tokens
        )
        t_context = (time.perf_counter() - t0) * 1000

        # Số chunks thực tế (dùng để giới hạn citation range [1]-[N])
        num_chunks = len(merged_chunks)
        print(f"[RAG STEP 4] 📄 Đóng gói Ngữ cảnh (Context Builder) ... [Xong: {t_context:.1f} ms | Đã dùng: {num_chunks} chunks]")

        # 8. Format sources for UI
        sources = format_sources_for_ui(merged_chunks)

        # 9. Relevance check
        if not formatted_context or formatted_context.strip() == "":
            no_result_msg = self._get_no_result_message(intent)
            print("[RAG PIPELINE] ⚠️ Không có ngữ cảnh phù hợp để trả lời.")
            return {
                "stream": iter([no_result_msg]),
                "sources": sources,
                "no_context": True
            }

        if not has_sufficient_relevance(merged_chunks):
            no_result_msg = self._get_no_result_message(intent)
            print("[RAG PIPELINE] ⚠️ Độ liên quan của tài liệu chưa đạt ngưỡng.")
            return {
                "stream": iter([no_result_msg]),
                "sources": sources,
                "no_context": True
            }

        # 10. Select prompt
        if intent["is_summary"]:
            prompt_template = SUMMARY_PROMPT
        else:
            prompt_template = SYSTEM_PROMPT

        system_content = _format_prompt(prompt_template, formatted_context, num_chunks)

        # 11. Build messages
        messages = [{"role": "system", "content": system_content}]
        if chat_history:
            messages.extend(chat_history[-4:])
        final_query = user_query if enable_thinking else f"/no_think {user_query}"
        messages.append({"role": "user", "content": final_query})

        # 12. Stream LLM
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
