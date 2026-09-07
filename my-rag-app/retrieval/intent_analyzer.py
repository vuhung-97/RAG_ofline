"""IntentAnalyzer — SRP: phân tích ý định (intent) của câu hỏi người dùng.

Sửa: Extract chapter/section cho MỌI query, không chỉ "tóm tắt".
"""

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
            "section_match": None,
            "article_match": None,
            "appendix_match": None,
        }

        # 1. Summary detection
        if q_low.startswith(("/tomtat", "/tonghop")):
            intent["is_summary"] = True
        elif re.search(r"tóm tắt|tom tat|tổng hợp|tong hop", user_query, re.I):
            intent["is_summary"] = True

        # 2. Chapter extraction — cho MỌI query (không chỉ summary)
        # "Chương 3", "Ch.3", "CHƯƠNG III", "chuong 3"
        m = re.search(r"ch(?:ương)?\.?\s*(\d+|[IVXLC]+)", user_query, re.I)
        if m:
            intent["chapter_match"] = m.group(1)

        # 3. Section extraction
        # "Mục 2.1", "Muc 2.1", "§ 2.1"
        m = re.search(r"(?:mục|muc|§)\s*([\d\.]+)", user_query, re.I)
        if m:
            intent["section_match"] = m.group(1)

        # 4. Article extraction
        # "Điều 5", "Điều 12", "dieu 5"
        m = re.search(r"điều\s*(\d+)", user_query, re.I)
        if m:
            intent["article_match"] = m.group(1)

        # 5. Appendix extraction
        # "Phụ lục A", "Phu luc A"
        m = re.search(r"phụ lục\s*([A-Z])", user_query, re.I)
        if m:
            intent["appendix_match"] = m.group(1)

        return intent
