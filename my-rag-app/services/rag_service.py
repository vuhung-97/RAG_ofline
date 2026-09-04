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

CITATION_FOOTER = """

Trích nguồn:
{citations}"""

SYSTEM_PROMPT = """Bạn là trợ lý tra cứu tài liệu. Trả lời câu hỏi dựa trên ngữ cảnh.

NGUYÊN TẮC TUYỆT ĐỐI:
- CHỈ dùng thông tin CÓ TRONG Context bên dưới.
- KHÔNG bịa đặt, KHÔNG thêm kiến thức ngoài Context.
- Nếu Context KHÔNG có thông tin → trả lời NGAY: "Tài liệu không đề cập đến thông tin này."
- Chỉ được trích nguồn trong khoảng [1]-[{num_chunks}].

CẤU TRÚC CÂU TRẢ LỜI:
- Mỗi ý chính xuống dòng mới.
- Dùng bullet (-) hoặc numbered list (1. 2. 3.).
- Mỗi thông tin PHẢI có trích nguồn [1][2] ngay sau.
- Trích nguồn CUỐI cùng, tách riêng khỏi câu trả lời bởi 2 dòng trống.

ĐỊNH DẠNG TRÍCH NGUỒN:
Kết thúc mỗi câu trả lời PHẢI có đoạn:
Trích nguồn:
[1] (Tài liệu: {{file_name}} | Chương: {{chapter}} | Mục: {{heading}}) "mô tả nội dung ngắn..."

VÍ DỤ CÂU TRẢ LỜI:
Bảng NGUOIDUNG có các thuộc tính sau:
1. id_nguoidung - Mã người dùng (C(6), Số nguyên) [1]
2. tennguoidung - Tên người dùng (C(50), Chữ cái) [1]
3. matkhau - Mật khẩu (C(50), Chữ cái) [1]

Bộ phận tham gia:
- Bộ phận bán hàng (BP02) [2]
- Bộ phận tài chính (BP03) [2]
{footer}

Ngữ cảnh:
{context}"""

SUMMARY_PROMPT = """Bạn là trợ lý tóm tắt tài liệu. Tóm tắt dựa trên ngữ cảnh.

NGUYÊN TẮC TUYỆT ĐỐI:
- CHỈ tóm tắt nội dung CÓ TRONG Context bên dưới.
- KHÔNG bịa đặt, KHÔNG thêm thông tin ngoài Context.
- Tóm tắt 100-300 từ, có cấu trúc bullet points.
- Mỗi bullet PHẢI trích nguồn [1][2]. Chỉ dùng [1]-[{num_chunks}].
- Nếu Context KHÔNG có thông tin → ghi: "Phần này không có trong tài liệu."

CẤU TRÚC CÂU TRẢ LỜI:
- Mỗi ý chính dùng bullet (-).
- Mỗi bullet trích nguồn [1][2] ở cuối dòng.
- Xuống dòng rõ ràng, dễ đọc.
- Trích nguồn CUỐI cùng, tách riêng khỏi câu trả lời bởi 2 dòng trống.

ĐỊNH DẠNG TRÍCH NGUỒN:
Kết thúc mỗi câu trả lời PHẢI có đoạn:
Trích nguồn:
[1] (Tài liệu: {{file_name}} | Chương: {{chapter}} | Mục: {{heading}}) "mô tả nội dung ngắn..."

VÍ DỤ CÂU TRẢ LỜI:
Quy trình QT02 là quy trình bán hàng cho khách hàng [1]:
- Quy trình bao gồm 3 bước chính [1]
- Bước 1: Tiếp nhận hàng hóa từ kho [2]
- Bước 2: Giao hàng cho khách hàng [2]
- Bước 3: Lập hóa đơn bán hàng [2]
{footer}

Ngữ cảnh:
{context}"""

EXTRACT_PROMPT = """Bạn là trợ lý trích xuất nguyên văn. Copy NGUYÊN VĂN từ ngữ cảnh.

NGUYÊN TẮC TUYỆT ĐỐI:
- CHỈ copy những gì CÓ TRONG Context bên dưới.
- KHÔNG tạo nội dung mới, KHÔNG diễn giải.
- Giữ nguyên định dạng (bảng, xuống dòng, bullet).
- Trích nguồn [idx] ở cuối mỗi dòng/thông tin. Chỉ dùng [1]-[{num_chunks}].
- Nếu Context KHÔNG có → nói: "Tài liệu không đề cập đến thông tin này."
- Trích nguồn CUỐI cùng, tách riêng khỏi nội dung trích bởi 2 dòng trống.

ĐỊNH DẠNG TRÍCH NGUỒN:
Kết thúc mỗi câu trả lời PHẢI có đoạn:
Trích nguồn:
[1] (Tài liệu: {{file_name}} | Chương: {{chapter}} | Mục: {{heading}}) "mô tả nội dung ngắn..."

VÍ DỤ CÂU TRẢ LỜI:
id_nguoidung | tennguoidung | matkhau [1]
001 | Nguyễn Văn A | ***** [1]
002 | Trần Văn B | ***** [1]
{footer}

Ngữ cảnh:
{context}"""


def _format_prompt(template: str, context: str, num_chunks: int) -> str:
    """Format prompt template bằng cách chèn CITATION_FOOTER vào {footer}."""
    footer = CITATION_FOOTER.format(citations="")
    return template.format(context=context, num_chunks=num_chunks, footer=footer)


def _postprocess_response(text: str) -> str:
    """Đảm bảo phần Trích nguồn không bị lặp và tách biệt khỏi câu trả lời."""
    parts = text.split("Trích nguồn:")
    if len(parts) > 2:
        answer = parts[0].rstrip()
        citations = "Trích nguồn:" + parts[-1]
        text = f"{answer}\n\n{citations}"
    return text


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

    def _validate_answer(self, answer: str, context: str) -> str:
        """Guardrail: Kiểm tra hallucination bằng word overlap + length check."""
        answer_lower = answer.lower().strip()

        # Always allow fallback phrases
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

        # Nếu answer có trích nguồn [1], [2]... → cho qua (đang dùng context)
        if _re.search(r'\[\d+\]', answer):
            return answer

        # No context → must reject
        if not context or context.strip() == "Không tìm thấy ngữ cảnh trong nhóm tài liệu hiện tại.":
            return "Tài liệu không đề cập đến thông tin này."

        # Length sanity check: answer > 3x context length → likely hallucinated
        if len(answer) > len(context) * 3 and len(answer) > 500:
            return "Tài liệu không đề cập đến thông tin này."

        # Word overlap check with higher threshold
        answer_words = set(_re.findall(r'\w{4,}', answer_lower))
        context_words = set(_re.findall(r'\w{4,}', context.lower()))

        if answer_words:
            overlap = len(answer_words & context_words) / len(answer_words)
            if overlap < 0.3:
                return "Tài liệu không đề cập đến thông tin này."

        # Check for new domain keywords not in context (NER-like)
        domain_keywords = [
            "mysql", "postgresql", "oracle", "sql server",
            "python", "java", "javascript", "php",
            "react", "angular", "vue",
            "windows", "linux", "macos",
            "cloud", "aws", "azure", "google cloud"
        ]
        for kw in domain_keywords:
            if kw in answer_lower and kw not in context.lower():
                return "Tài liệu không đề cập đến thông tin này."

        return answer

    def _detect_intent(self, user_query: str) -> Dict[str, Any]:
        """Phân tích intent của user query."""
        q_low = user_query.strip().lower()
        intent = {
            "is_extract": False,
            "is_summary": False,
            "chapter_match": None,
        }

        if q_low.startswith(("/trich", "/extract")):
            intent["is_extract"] = True
        elif q_low.startswith(("/tomtat", "/tonghop")):
            intent["is_summary"] = True
            m = _re.search(r"chương\s*(\d+)|chuong\s*(\d+)", user_query, _re.I)
            if m:
                intent["chapter_match"] = m.group(1) or m.group(2)
        else:
            if _re.search(r"trích.*nguyên văn|nguyên văn|trích.*đoạn|bảng\s*\d+|extract|verbatim", user_query, _re.I):
                intent["is_extract"] = True
            elif _re.search(r"tóm tắt|tom tat|tổng hợp|tong hop", user_query, _re.I):
                m = _re.search(r"chương\s*(\d+)|chuong\s*(\d+)|mục\s*([\d\.]+)|muc\s*([\d\.]+)", user_query, _re.I)
                if m:
                    intent["chapter_match"] = m.group(1) or m.group(2) or m.group(3) or m.group(4)
                intent["is_summary"] = True

        return intent

    def _get_all_chunks_map(self) -> Dict[str, Dict[str, Any]]:
        """Lấy tất cả chunks từ ChromaDB để build neighbor map."""
        chunks_map = {}
        try:
            all_data = self.vector_store.collection.get()
            if all_data and all_data["ids"]:
                for cid, doc, meta in zip(all_data["ids"], all_data["documents"], all_data["metadatas"]):
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

        # Boost tables for extract intent with "bảng"
        if intent.get("is_extract") and _re.search(r"bảng|table", user_query, _re.I):
            for r in fused:
                if r.get("metadata", {}).get("type") == "table":
                    r["rrf_score"] = r.get("rrf_score", 0) + 0.005

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

        # 1. Intent detection
        intent = self._detect_intent(user_query)

        # 2. Embed query với prefix
        query_vector = self.embedding_service.embed_query(user_query, model_name=embed_model)

        # 3. Hybrid search (Semantic + BM25 + RRF)
        fused_results = self._hybrid_search(
            user_query, query_vector,
            semantic_top_k=top_k,
            bm25_top_k=top_k,
            final_top_k=top_k,
            intent=intent
        )

        # 4. Dedup
        deduped = deduplicate_chunks(fused_results)

        # 5. Neighbor expansion
        if config.ENABLE_NEIGHBOR_EXPANSION and deduped:
            all_chunks_map = self._get_all_chunks_map()
            expanded = expand_neighbors(
                deduped, all_chunks_map,
                max_expansion=config.NEIGHBOR_MAX_EXPANSION,
                same_section_only=config.NEIGHBOR_SAME_SECTION_ONLY
            )
        else:
            expanded = deduped

        # 6. Context budget (neighbors đã merge vào chunk gốc → số chunks = top_k)
        final_chunks = expanded[:top_k]

        # 7. Format context (build_context tự merge table chunks bên trong)
        formatted_context, merged_chunks = build_context(
            final_chunks,
            max_chunks=top_k,
            max_tokens=config.MAX_CONTEXT_TOKENS
        )

        # Số chunks thực tế (dùng để giới hạn citation range [1]-[N])
        num_chunks = len(merged_chunks)

        # 8. Format sources for UI (= merged chunks, không cần cap)
        sources = format_sources_for_ui(merged_chunks)

        # 9. Relevance check
        if not formatted_context or formatted_context.strip() == "":
            no_result_msg = self._get_no_result_message(intent)
            return {
                "stream": iter([no_result_msg]),
                "sources": sources,
                "no_context": True
            }

        if not has_sufficient_relevance(fused_results):
            no_result_msg = self._get_no_result_message(intent)
            return {
                "stream": iter([no_result_msg]),
                "sources": sources,
                "no_context": True
            }

        # 10. Select prompt
        if intent["is_extract"]:
            prompt_template = EXTRACT_PROMPT
        elif intent["is_summary"]:
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
        stream_generator = self.llm_service.stream_chat(
            messages=messages,
            model_name=llm_model,
            num_ctx=num_ctx,
            temperature=temperature,
            enable_thinking=enable_thinking
        )

        return {
            "stream": stream_generator,
            "sources": sources
        }

    @staticmethod
    def _get_no_result_message(intent: Dict[str, Any]) -> str:
        """Trả về message phù hợp khi không tìm thấy kết quả."""
        if intent.get("is_extract"):
            return "Không tìm thấy nội dung phù hợp trong tài liệu được cung cấp."
        else:
            return "Không tìm thấy thông tin phù hợp trong tài liệu được cung cấp."
