"""Chat Logger - Ghi log câu hỏi & câu trả lời để phân tích."""

import json
import os
from datetime import datetime
from typing import List, Dict, Any


class ChatLogger:
    """Ghi log mỗi lượt chat ra file JSONL (mỗi dòng = 1 entry)."""

    def __init__(self, log_dir: str = None):
        if log_dir is None:
            log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

    def _log_file(self) -> str:
        """Trả về path file log theo ngày: logs/session_YYYY-MM-DD.jsonl"""
        date_str = datetime.now().strftime("%Y-%m-%d")
        return os.path.join(self.log_dir, f"session_{date_str}.jsonl")

    def log_entry(
        self,
        query: str,
        response: str,
        sources: List[Dict[str, Any]] = None,
        settings: Dict[str, Any] = None,
        no_context: bool = False,
        elapsed_ms: int = 0,
        error: str = None
    ):
        """Ghi 1 dòng log vào file JSONL."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "response": response,
            "sources_count": len(sources) if sources else 0,
            "sources": self._trim_sources(sources) if sources else [],
            "settings": settings or {},
            "no_context": no_context,
            "elapsed_ms": elapsed_ms,
        }
        if error:
            entry["error"] = error

        try:
            with open(self._log_file(), "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _trim_sources(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rút gọn sources để log (chỉ giữ metadata, không lưu full text)."""
        trimmed = []
        for s in sources:
            trimmed.append({
                "file_name": s.get("file_name", ""),
                "distance": s.get("distance", 0),
                "rerank_score": s.get("rerank_score"),
                "text_preview": s.get("text", "")[:150]
            })
        return trimmed

    def get_log_count(self) -> int:
        """Trả về số entries trong ngày."""
        log_file = self._log_file()
        if not os.path.exists(log_file):
            return 0
        with open(log_file, "r", encoding="utf-8") as f:
            return sum(1 for _ in f)
