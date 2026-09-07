"""Query Rewriter — SRP: Detect follow-up queries và rewrite thành standalone.

Sử dụng regex/heuristic, KHÔNG dùng LLM cho bước này.
"""

import re
from typing import List, Dict, Any


class FollowUpDetector:
    """Phát hiện query có phải follow-up không."""

    # Các từ thường bắt đầu follow-up query (tiếng Việt)
    FOLLOW_UP_STARTERS = re.compile(
        r'^(còn|thì|vậy|thế|như|giống|khác|bên|ngoài|trong|đó|này|kia|hay|hay là|còn như|thêm vào|liệu)\s',
        re.I
    )

    # Ngưỡng độ dài query ngắn → có thể là follow-up
    SHORT_QUERY_THRESHOLD = 25

    @staticmethod
    def is_follow_up(query: str, chat_history: List[Dict[str, str]] = None) -> bool:
        """Kiểm tra query có phải follow-up không."""
        if not chat_history:
            return False

        q = query.strip()

        # Bắt đầu bằng pronoun/starter → follow-up
        if FollowUpDetector.FOLLOW_UP_STARTERS.match(q):
            return True

        # Query rất ngắn + KHÔNG có cấu trúc câu đầy đủ → follow-up
        # (VD: "KQ?", "thế nào?", "Còn sao?")
        if len(q) < FollowUpDetector.SHORT_QUERY_THRESHOLD:
            # Chỉ follow-up nếu query không chứa verb/noun indicator
            has_structure = bool(re.search(
                r'(điều|mục|chương|quy|định|chính|sách|quy tắc|quy định|trách nhiệm|nghĩa vụ|cho|của)',
                q, re.I
            ))
            if not has_structure:
                return True

        return False


class QueryRewriter:
    """Rewrite follow-up query thành standalone retrieval query."""

    @staticmethod
    def rewrite(query: str, chat_history: List[Dict[str, str]] = None) -> str:
        """Rewrite follow-up query thành standalone query.

        Strategy: Extract topic từ last user message, prepend vào query nếu cần.
        KHÔNG thêm kiến thức, KHÔNG đoán câu trả lời.
        """
        if not chat_history:
            return query

        if not FollowUpDetector.is_follow_up(query, chat_history):
            return query

        # Lấy topic từ last user message
        last_user = chat_history[-1].get("content", "")
        topic = QueryRewriter._extract_topic(last_user)

        if topic:
            # Thêm topic vào query để standalone
            rewritten = f"{query} về {topic}"
            return rewritten

        return query

    @staticmethod
    def _extract_topic(message: str) -> str:
        """Extract topic/main noun phrase từ message."""
        q = message.strip()

        # Pattern: "Chương X về Y" → Y
        m = re.search(r'ch(?:ương)?\.?\s*\d+.*?(?:về|liên quan đến|nói về)\s+(.{5,50})', q, re.I)
        if m:
            return m.group(1).strip().rstrip('?.,;:')

        # Pattern: "Điều X về Y" → Y
        m = re.search(r'điều\s*\d+.*?(?:về|liên quan đến|nói về)\s+(.{5,50})', q, re.I)
        if m:
            return m.group(1).strip().rstrip('?.,;:')

        # Pattern: "Quy định về Y là gì?" → Y
        m = re.search(r'(?:quy định|chính sách|quy tắc|nội quy)\s+(?:về|liên quan đến)?\s*(.{5,50})\s*(?:là gì|thế nào|như thế nào|\?)', q, re.I)
        if m:
            return m.group(1).strip().rstrip('?.,;:')

        # Fallback: lấy 3-5 từ đầu có nghĩa
        words = q.split()
        if len(words) >= 3:
            # Bỏ các từ nhỏ ở đầu
            meaningful = [w for w in words if len(w) > 2]
            if meaningful:
                return " ".join(meaningful[:5])

        return ""
