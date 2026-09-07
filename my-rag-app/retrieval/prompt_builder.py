"""PromptBuilder — SRP: quản lý và đóng gói các mẫu Prompt cho LLM.

Sửa: Prompt ngắn gọn hơn cho instruct model, bỏ logic thinking.
"""

from typing import List, Dict, Any
from config import config

SYSTEM_PROMPT = """Bạn là trợ lý tra cứu tài liệu. Trả lời dựa trên Context bên dưới.

NGUYÊN TẮC:
- CHỈ dùng thông tin CÓ TRONG Context.
- KHÔNG bịa đặt, KHÔNG thêm kiến thức ngoài Context.
- Nếu Context KHÔNG có thông tin → trả lời: "Tài liệu không đề cập đến thông tin này."
- Gắn nhãn nguồn [1], [2] ở cuối mỗi ý. Chỉ dùng nhãn từ [1] đến [{num_chunks}].
- Không viết mục "Citations:", "Tài liệu tham khảo:" ở cuối.
- Trả lời trực tiếp, rõ ràng.

Ngữ cảnh:
{context}"""

SUMMARY_PROMPT = """Bạn là trợ lý tóm tắt tài liệu. Tóm tắt dựa trên Context bên dưới.

NGUYÊN TẮC:
- CHỈ tóm tắt nội dung CÓ TRONG Context.
- KHÔNG bịa đặt, KHÔNG thêm thông tin ngoài Context.
- Nếu Context KHÔNG có thông tin → ghi: "Phần này không có trong tài liệu."
- Nhãn nguồn đặt ở cuối mỗi dòng bullet. Chỉ dùng nhãn từ [1] đến [{num_chunks}].
- Không viết mục "Citations:", "Tài liệu tham khảo:" ở cuối.

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
        enable_thinking: bool = False
    ) -> List[Dict[str, str]]:
        """Đóng gói danh sách messages gửi cho LLM (gồm system prompt, history, user query)."""
        messages = [{"role": "system", "content": system_content}]
        limit = config.CHAT_HISTORY_LIMIT if hasattr(config, "CHAT_HISTORY_LIMIT") else 4
        if chat_history:
            messages.extend(chat_history[-limit:])

        messages.append({"role": "user", "content": user_query})
        return messages
