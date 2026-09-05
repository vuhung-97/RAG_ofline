"""PromptBuilder — SRP: quản lý và đóng gói các mẫu Prompt cho LLM."""

from typing import List, Dict, Any
from config import config

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


class PromptBuilder:
    """Quản lý các template prompt và đóng gói danh sách messages gửi cho LLM."""

    @staticmethod
    def build_system_content(is_summary: bool, formatted_context: str, num_chunks: int) -> str:
        """Chọn và format system prompt template."""
        template = SUMMARY_PROMPT if is_summary else SYSTEM_PROMPT
        return template.format(context=formatted_context, num_chunks=num_chunks)

    @staticmethod
    def build_messages(
        system_content: str,
        user_query: str,
        chat_history: List[Dict[str, str]] = None,
        enable_thinking: bool = True
    ) -> List[Dict[str, str]]:
        """Đóng gói danh sách messages gửi cho LLM (gồm system prompt, history, user query)."""
        messages = [{"role": "system", "content": system_content}]
        limit = config.CHAT_HISTORY_LIMIT if hasattr(config, "CHAT_HISTORY_LIMIT") else 4
        if chat_history:
            messages.extend(chat_history[-limit:])

        final_query = user_query if enable_thinking else f"/no_think {user_query}"
        messages.append({"role": "user", "content": final_query})
        return messages
