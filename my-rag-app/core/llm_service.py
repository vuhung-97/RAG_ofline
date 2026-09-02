from typing import List, Dict, Any, Generator
import json
import requests
from config import config

class OllamaLLMService:
    """SRP: Tương tác với Ollama LLM (Qwen3:1.7b) với hỗ trợ Generator Streaming."""

    def __init__(self, model_name: str = config.LLM_MODEL, host: str = config.OLLAMA_HOST):
        self.model_name = model_name
        self.api_url = f"{host}/api/chat"

    def stream_chat(self, messages: List[Dict[str, str]]) -> Generator[str, None, None]:
        """Gửi prompt tới LLM và stream câu trả lời từng từ một."""
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": True,
            "keep_alive": config.OLLAMA_KEEP_ALIVE,
            "options": {
                "num_ctx": config.LLM_NUM_CTX
            }
        }

        try:
            response = requests.post(self.api_url, json=payload, stream=True, timeout=120)
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line.decode("utf-8"))
                    if "message" in chunk and "content" in chunk["message"]:
                        yield chunk["message"]["content"]
        except requests.exceptions.RequestException as e:
            yield f"\n[Lỗi kết nối Ollama LLM ({self.model_name}): {str(e)}]"
