from typing import List, Dict, Any, Generator
from core.embedding_service import OllamaEmbeddingService
from core.vector_store import ChromaVectorStore
from core.llm_service import OllamaLLMService
from config import config

SYSTEM_PROMPT = """Bạn là trợ lý ảo tra cứu tài liệu chuyên nghiệp. 
Nhiệm vụ của bạn là trả lời câu hỏi của người dùng dựa TRỰC TIẾP trên các đoạn ngữ cảnh (Context) được cung cấp dưới đây.

Quy tắc quan trọng:
1. Chỉ sử dụng thông tin trong phần Ngữ cảnh được cung cấp để trả lời.
2. Nếu ngữ cảnh không chứa thông tin để trả lời, hãy lịch sự thông báo "Tôi không tìm thấy thông tin phù hợp trong tài liệu được cung cấp." và không tự bịa thông tin.
3. Trả lời bằng tiếng Việt rõ ràng, ngắn gọn, súc tích và mạch lạc.

Ngữ cảnh được cung cấp:
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
        temperature: float = config.TEMPERATURE
    ) -> Dict[str, Any]:
        """Xử lý câu hỏi: Tra cứu vector trong workspace hiện tại và stream câu trả lời."""
        # 1. Embed query câu hỏi
        query_vector = self.embedding_service.embed_text(user_query, model_name=embed_model)

        # 2. Tìm kiếm Top-K đoạn tương đồng trong Active Workspace
        search_results = self.vector_store.search_similarity(query_vector, top_k=top_k)

        # 3. Dựng context
        context_parts = []
        sources = []
        for idx, res in enumerate(search_results, 1):
            file_name = res["metadata"].get("file_name", "Unknown")
            context_parts.append(f"[{idx}] (Nguồn: {file_name}):\n{res['text']}")
            sources.append({
                "file_name": file_name,
                "text": res["text"]
            })

        formatted_context = "\n\n".join(context_parts) if context_parts else "Không tìm thấy ngữ cảnh trong nhóm tài liệu hiện tại."
        system_content = SYSTEM_PROMPT.format(context=formatted_context)

        # 4. Xây dựng tin nhắn gửi LLM
        messages = [{"role": "system", "content": system_content}]
        if chat_history:
            messages.extend(chat_history[-4:])
        messages.append({"role": "user", "content": user_query})

        # 5. Stream kết quả từ LLM với thông số động
        stream_generator = self.llm_service.stream_chat(
            messages=messages,
            model_name=llm_model,
            num_ctx=num_ctx,
            temperature=temperature
        )

        return {
            "stream": stream_generator,
            "sources": sources
        }
