"""LLM Reranker dùng Ollama (model người dùng chọn) - SRP: chấm điểm lại."""

import re
import json
import time
import requests
from typing import List, Dict, Any
from config import config


RERANK_PROMPT = """Bạn là chuyên gia đánh giá độ liên quan.
Câu hỏi: "{query}"

Hãy chấm điểm TỪNG ĐOẠN dưới đây theo độ liên quan để trả lời câu hỏi, thang 0-10 (10 rất liên quan).
Phải chấm ĐIỂM cho TẤT CẢ các đoạn, không được bỏ sót.

Trả về JSON CHÍNH XÁC theo format:
{{"scores": [{{"id": 1, "score": 5}}, {{"id": 2, "score": 7}}]}}

Với:
- id: số thứ tự của đoạn (1, 2, 3, ...)
- score: điểm từ 0 đến 10

KHÔNG giải thích. CHỈ trả về JSON.

Các đoạn:
{docs}
"""

SYSTEM_PROMPT = "Bạn chỉ trả về JSON scores."


class LLMReranker:
    """Rerank bằng Ollama LLM - 1 call cho tất cả candidates."""

    def __init__(self, host: str = config.OLLAMA_HOST):
        self.host = host
        self.api_url = f"{host}/api/chat"
        self.timeout = config.RERANK_TIMEOUT
        self.keep_alive = config.OLLAMA_KEEP_ALIVE

    def rerank(self, query: str, candidates: List[Dict[str, Any]], model_name: str, top_k: int = 6) -> List[Dict[str, Any]]:
        if not candidates:
            return candidates

        # Gán score mặc định rồi trả về nếu không cần gọi LLM
        # if len(candidates) <= top_k:
        #     for c in candidates:
        #         c["rerank_score"] = c.get("distance", 0)
        #     return candidates

        t0 = time.perf_counter()

        # Dựng docs list - lấy full text, không cắt ngắn
        docs_text = ""
        for idx, c in enumerate(candidates, 1):
            text = c.get("text", "").replace("\n", " ")
            docs_text += f"[{idx}] {text}\n"

        prompt = RERANK_PROMPT.format(query=query, docs=docs_text)

        # Debug: in prompt
        print(f"\n[RERANK DEBUG] Prompt ({len(prompt)} chars, ~{len(prompt)//4} tokens):")
        print(f"---\n{prompt[:500]}{'...' if len(prompt) > 500 else ''}\n---")

        # Tự tính num_ctx dựa trên độ dài thực tế
        total_chars = len(SYSTEM_PROMPT) + len(prompt)
        estimated_tokens = total_chars // 4
        num_ctx = max(2048, 1 << (estimated_tokens - 1).bit_length())  # round up to power of 2

        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "think": False,
            "keep_alive": self.keep_alive,
            "options": {"temperature": 0.1, "num_ctx": num_ctx}
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

            # Debug: in raw response
            print(f"\n[RERANK DEBUG] Raw response ({len(content)} chars):")
            print(f"---\n{content}\n---")

            scores = self._parse_scores(content, len(candidates))

            # Gán score cho từng candidate
            for idx, c in enumerate(candidates):
                c["rerank_score"] = scores.get(idx + 1, 0)

            # Sort giảm dần theo rerank_score, tie-breaker = distance nhỏ hơn
            reranked = sorted(candidates, key=lambda x: (-x.get("rerank_score", 0), x.get("distance", 999)))

            t_rerank = (time.perf_counter() - t0) * 1000
            print(f"[RERANK] Hoàn thành: {t_rerank:.1f} ms | {len(candidates)} candidates → top_{top_k} | num_ctx={num_ctx}")

            return reranked[:top_k]

        except Exception as e:
            t_rerank = (time.perf_counter() - t0) * 1000
            print(f"[WARN] Rerank lỗi ({t_rerank:.1f} ms): {e}")
            # Fallback: giữ thứ tự cũ
            for c in candidates:
                c["rerank_score"] = 0
            return candidates[:top_k]

    def _parse_scores(self, content: str, n: int) -> Dict[int, float]:
        scores: Dict[int, float] = {}
        if not content:
            return scores

        # Thử parse JSON { "scores": [{"id": 1, "score": 8}, ...] }
        try:
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

        # Fallback regex: [1]: 8 hoặc "1": 7.5
        for match in re.finditer(r"\[?\s*(\d+)\s*\]?\s*[:\-]\s*(\d+(?:\.\d+)?)", content):
            try:
                idx = int(match.group(1))
                sc = float(match.group(2))
                if 1 <= idx <= n and idx not in scores:
                    scores[idx] = sc
            except Exception:
                continue

        # Fallback: "id 1 score 8"
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
