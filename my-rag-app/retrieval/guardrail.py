"""GuardrailValidator — SRP: kiểm tra và lọc ảo giác (hallucination) trong câu trả lời LLM."""

import re
from config import config

FALLBACK_PHRASES = [
    "tài liệu không đề cập",
    "không tìm thấy thông tin",
    "không có nội dung",
    "không tìm thấy vị trí",
    "phần này không có trong tài liệu",
    "không biết",
    "không chắc"
]


class GuardrailValidator:
    """Kiểm tra tính hợp lệ của câu trả lời từ LLM trước khi chốt hiển thị."""

    @staticmethod
    def validate_answer(answer: str, context: str, num_chunks: int = 0) -> str:
        """Kiểm tra hallucination dựa trên citations, length ratio và word overlap."""
        # Loại bỏ <think>...</think> nếu còn sót
        answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
        answer_lower = answer.lower().strip()

        # 1. Fallback phrases → cho qua
        if any(p in answer_lower for p in FALLBACK_PHRASES):
            return answer

        # 2. Citation hợp lệ [1]-[num_chunks] → cho qua
        if num_chunks > 0:
            citations = re.findall(r'\[(\d+)\]', answer)
            if citations:
                max_cite = max(int(c) for c in citations)
                if max_cite <= num_chunks:
                    return answer

        # 3. Không có context → reject
        if not context or context.strip() == "":
            return "Tài liệu không đề cập đến thông tin này."

        # 4. Length check: answer > 3x context → reject
        if len(answer) > len(context) * 3 and len(answer) > 500:
            return "Tài liệu không đề cập đến thông tin này."

        # 5. Word overlap check (chỉ áp dụng cho answer dài)
        if len(answer) > 100:
            answer_words = set(re.findall(r'\w{4,}', answer_lower))
            context_words = set(re.findall(r'\w{4,}', context.lower()))
            if answer_words:
                overlap = len(answer_words & context_words) / len(answer_words)
                min_overlap = config.GUARDRAIL_MIN_WORD_OVERLAP if hasattr(config, "GUARDRAIL_MIN_WORD_OVERLAP") else 0.3
                if overlap < min_overlap:
                    return "Tài liệu không đề cập đến thông tin này."

        return answer
