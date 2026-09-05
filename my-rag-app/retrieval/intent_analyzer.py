"""IntentAnalyzer — SRP: phân tích ý định (intent) của câu hỏi người dùng."""

import re
from typing import Dict, Any


class IntentAnalyzer:
    """Phân tích intent câu hỏi (ví dụ tóm tắt, tổng hợp, trích xuất chương/mục)."""

    @staticmethod
    def detect_intent(user_query: str) -> Dict[str, Any]:
        """Phân tích intent của câu hỏi."""
        q_low = user_query.strip().lower()
        intent = {
            "is_summary": False,
            "chapter_match": None,
        }

        if q_low.startswith(("/tomtat", "/tonghop")):
            intent["is_summary"] = True
            m = re.search(r"chương\s*(\d+)|chuong\s*(\d+)", user_query, re.I)
            if m:
                intent["chapter_match"] = m.group(1) or m.group(2)
        elif re.search(r"tóm tắt|tom tat|tổng hợp|tong hop", user_query, re.I):
            m = re.search(r"chương\s*(\d+)|chuong\s*(\d+)|mục\s*([\d\.]+)|muc\s*([\d\.]+)", user_query, re.I)
            if m:
                intent["chapter_match"] = m.group(1) or m.group(2) or m.group(3) or m.group(4)
            intent["is_summary"] = True

        return intent
