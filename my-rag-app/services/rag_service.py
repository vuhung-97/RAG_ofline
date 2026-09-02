from typing import List, Dict, Any, Generator
from core.embedding_service import OllamaEmbeddingService
from core.vector_store import ChromaVectorStore
from core.llm_service import OllamaLLMService
from config import config

SYSTEM_PROMPT = """Bạn là trợ lý ảo tra cứu tài liệu chuyên nghiệp.
Nhiệm vụ của bạn là trả lời câu hỏi dựa TRỰC TIẾP trên các đoạn ngữ cảnh (Context) kèm điểm tin cậy được cung cấp dưới đây.

QUY TRÌNH BẮT BUỘC trước khi trả lời:
1. Đánh giá từng đoạn [1]..[N]: ghi nhận liên quan / không liên quan / mâu thuẫn.
2. Chỉ trả lời khi có ít nhất 2 đoạn ủng hộ hoặc 1 đoạn rất khớp (điểm cao); nếu các đoạn mâu thuẫn hãy nêu cả hai quan điểm.
3. Mỗi khẳng định quan trọng phải trích nguồn [1][2] tương ứng.
4. Nếu ngữ cảnh không đủ bằng chứng, hãy nói rõ "Tôi không tìm thấy thông tin phù hợp về X trong tài liệu được cung cấp." và liệt kê thiếu gì, không tự bịa.
5. Trả lời bằng tiếng Việt rõ ràng, ngắn gọn, súc tích và mạch lạc, ưu tiên tổng hợp thay vì chỉ trích đoạn đầu tiên.

Ngữ cảnh được cung cấp (kèm điểm tin cậy):
{context}
"""

class RAGService:
    """SRP & Orchestration: Điều phối luồng Query với các thông số linh hoạt."""

    def __init__(
        self,
        embedding_service: OllamaEmbeddingService,
        vector_store: ChromaVectorStore,
        llm_service: OllamaLLMService
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.llm_service = llm_service

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
        """Xử lý câu hỏi: Tra cứu vector, rerank bằng LLM nếu bật, rồi stream."""
        # 1. Embed query câu hỏi
        query_vector = self.embedding_service.embed_text(user_query, model_name=embed_model)

        # 2. Tìm kiếm pool rộng hơn top_k để có nguyên liệu rerank
        pool_size = config.RERANK_CANDIDATES if enable_rerank else top_k
        search_results = self.vector_store.search_similarity(query_vector, top_k=pool_size)

        # 2b. Lọc ngưỡng distance để bỏ nhiễu xa
        if search_results and config.DISTANCE_THRESHOLD:
            filtered = [r for r in search_results if r.get("distance", 0) <= config.DISTANCE_THRESHOLD]
            if len(filtered) >= 2:
                search_results = filtered

        # 2c. Rerank bằng Ollama LLM nếu bật và đủ candidates
        if enable_rerank and len(search_results) > top_k:
            try:
                from core.reranker import LLMReranker
                reranker = LLMReranker()
                search_results = reranker.rerank(query=user_query, candidates=search_results, model_name=llm_model, top_k=top_k)
            except Exception:
                search_results = search_results[:top_k]
        else:
            search_results = search_results[:top_k]

        # 3. Dựng context kèm điểm tin cậy
        context_parts = []
        sources = []
        for idx, res in enumerate(search_results, 1):
            file_name = res["metadata"].get("file_name", "Unknown")
            score = res.get("rerank_score")
            dist = res.get("distance", 0)
            if score is not None:
                header = f"[{idx}] (Nguồn: {file_name} | điểm={score:.1f} | d={dist:.3f}):"
            else:
                header = f"[{idx}] (Nguồn: {file_name} | d={dist:.3f}):"
            context_parts.append(f"{header}\n{res['text']}")
            sources.append({
                "file_name": file_name,
                "text": res["text"],
                "rerank_score": score,
                "distance": dist
            })

        formatted_context = "\n\n".join(context_parts) if context_parts else "Không tìm thấy ngữ cảnh trong nhóm tài liệu hiện tại."
        system_content = SYSTEM_PROMPT.format(context=formatted_context)

        # 4. Xây dựng tin nhắn gửi LLM
        messages = [{"role": "system", "content": system_content}]
        if chat_history:
            messages.extend(chat_history[-4:])
        # fallback /no_think prefix khi tắt thinking để đảm bảo Qwen tuân thủ
        final_query = user_query if enable_thinking else f"/no_think {user_query}"
        messages.append({"role": "user", "content": final_query})

        # 5. Stream kết quả từ LLM với thông số động
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
