"""LLM Reranker dùng Ollama (model người dùng chọn) - SRP: chấm điểm lại."""

import re
import json
import requests
from typing import List, Dict, Any
from config import config


RERANK_PROMPT = """Bạn là chuyên gia đánh giá độ liên quan.
Câu hỏi: "{query}"

Hãy chấm điểm từng đoạn dưới đây theo độ liên quan để trả lời câu hỏi, thang 0-10 (10 rất liên quan).
Chỉ trả về JSON dạng: {{"scores": [{{"id": 1, "score": 8}}, {{"id": 2, "score": 3}}]}}
Không giải thích thêm.

Các đoạn:
{docs}
"""


class LLMReranker:
    """Rerank bằng Ollama LLM - 1 call cho tất cả candidates."""

    def __init__(self, host: str = config.OLLAMA_HOST, timeout: int = 25):
        self.host = host
        self.api_url = f"{host}/api/chat"
        self.timeout = timeout
        self.keep_alive = config.OLLAMA_KEEP_ALIVE

    def rerank(self, query: str, candidates: List[Dict[str, Any]], model_name: str, top_k: int = 6) -> List[Dict[str, Any]]:
        if not candidates:
            return candidates
        if len(candidates) <= top_k:
            # vẫn gán score mặc định để prompt hiển thị
            for c in candidates:
                c["rerank_score"] = c.get("distance", 0)
            return candidates

        # Dựng docs list
        docs_text = ""
        for idx, c in enumerate(candidates, 1):
            snippet = c.get("text", "")[:400].replace("\n", " ")
            docs_text += f"[{idx}] {snippet}\n"

        prompt = RERANK_PROMPT.format(query=query, docs=docs_text)
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "Bạn chỉ trả về JSON scores."},
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "think": False,
            "keep_alive": self.keep_alive,
            "options": {"temperature": 0.1, "num_ctx": 4096}
        }

        try:
            resp = requests.post(self.api_url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            content = ""
            if "message" in data and "content" in data["message"]:
                content = data["message"]["content"]
            elif "response" in data:
                content = data["response"]

            scores = self._parse_scores(content, len(candidates))
            # gắn score
            for idx, c in enumerate(candidates):
                # scores dict id->score
                c["rerank_score"] = scores.get(idx + 1, 0)
            # sort giảm dần theo rerank_score, tie-breaker distance nhỏ hơn
            reranked = sorted(candidates, key=lambda x: (-x.get("rerank_score", 0), x.get("distance", 999)))
            return reranked[:top_k]

        except Exception:
            # fallback: giữ thứ tự Chroma
            for c in candidates:
                c["rerank_score"] = 0
            return candidates[:top_k]

    def _parse_scores(self, content: str, n: int) -> Dict[int, float]:
        scores: Dict[int, float] = {}
        if not content:
            return scores
        # thử parse JSON
        try:
            # tìm JSON block {...}
            m = re.search(r"\{.*\}", content, flags=re.DOTALL)
            if m:
                obj = json.loads(m.group(0))
                if "scores" in obj and isinstance(obj["scores"], list):
                    for item in obj["scores"]:
                        try:
                            idx = int(item.get("id"))
                            sc = float(item.get("score"))
                            if 1 <= idx <= n:
                                scores[idx] = sc
                        except Exception:
                            continue
                    if scores:
                        return scores
        except Exception:
            pass

        # fallback regex: [1]: 8 hoặc "1": 7.5
        for match in re.finditer(r"\[?\s*(\d+)\s*\]?\s*[:\-]\s*(\d+(?:\.\d+)?)", content):
            try:
                idx = int(match.group(1))
                sc = float(match.group(2))
                if 1 <= idx <= n and idx not in scores:
                    scores[idx] = sc
            except Exception:
                continue

        # nếu vẫn rỗng, thử tìm dạng "id 1 score 8"
        if not scores:
            for match in re.finditer(r"id\s*(\d+).*?score\s*(\d+(?:\.\d+)?)", content, flags=re.IGNORECASE):
                try:
                    idx = int(match.group(1))
                    sc = float(match.group(2))
                    if 1 <= idx <= n and idx not in scores:
                        scores[idx] = sc
                except Exception:
                    continue

        return scores
