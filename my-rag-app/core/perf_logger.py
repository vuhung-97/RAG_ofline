"""Performance Logger — SRP: chỉ đo và log performance metrics."""

import os
import time
import json
import datetime
from typing import Dict, Any, Optional


class PerfLogger:
    """Measure and log timing for each RAG pipeline stage."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._stages: Dict[str, float] = {}
        self._start: Optional[float] = None

    def start(self):
        """Bắt đầu đo performance."""
        self._start = time.perf_counter()
        self._stages = {}

    def mark(self, stage: str):
        """Đánh dấu thời điểm hoàn thành 1 stage."""
        if self.enabled and self._start:
            self._stages[stage] = time.perf_counter()

    def get_metrics(self) -> Dict[str, Any]:
        """Trả về metrics dưới dạng dict (ms)."""
        if not self._start:
            return {}

        now = time.perf_counter()
        metrics = {}

        prev = self._start
        for stage, ts in sorted(self._stages.items(), key=lambda x: x[1]):
            metrics[f"t_{stage}_ms"] = round((ts - prev) * 1000, 1)
            prev = ts

        metrics["t_total_ms"] = round((now - self._start) * 1000, 1)
        return metrics

    def log(
        self,
        request_id: str,
        query: str,
        metrics: Dict[str, Any],
        extra: Dict[str, Any] = None
    ):
        """Log performance entry to file."""
        if not self.enabled:
            return

        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "request_id": request_id,
            "query": query[:200],
            "metrics": metrics
        }
        if extra:
            entry.update(extra)

        log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"perf_{datetime.date.today()}.jsonl")

        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass
