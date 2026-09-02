from typing import List, Dict, Any, Generator
import json
import re
import requests
from config import config

class OllamaLLMService:
    """SRP: Tương tác với Ollama LLM với tham số động (Model, num_ctx, temperature)."""

    def __init__(self, host: str = config.OLLAMA_HOST):
        self.host = host
        self.api_url = f"{host}/api/chat"

    def stream_chat(
        self,
        messages: List[Dict[str, str]],
        model_name: str = config.LLM_MODEL,
        num_ctx: int = config.LLM_NUM_CTX,
        temperature: float = config.TEMPERATURE,
        enable_thinking: bool = config.ENABLE_THINKING
    ) -> Generator[str, None, None]:
        """Gửi prompt tới LLM và stream câu trả lời từng từ một."""
        payload = {
            "model": model_name,
            "messages": messages,
            "stream": True,
            "think": enable_thinking,
            "keep_alive": config.OLLAMA_KEEP_ALIVE,
            "options": {
                "num_ctx": num_ctx,
                "temperature": temperature
            }
        }

        try:
            response = requests.post(self.api_url, json=payload, stream=True, timeout=120)
            response.raise_for_status()

            # Buffer để lọc <think> nếu server vẫn trả tag khi tắt thinking
            thinking_buffer = ""
            in_think = False
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line.decode("utf-8"))
                    # Ollama có thể trả field 'thinking' riêng
                    if not enable_thinking and "message" in chunk and chunk["message"].get("thinking"):
                        continue
                    if "message" in chunk and "content" in chunk["message"]:
                        content = chunk["message"]["content"]
                        if not enable_thinking and content:
                            # Lọc streaming <think>...</think> nếu còn sót
                            # đơn giản: nếu chứa tag thì buffer và strip
                            if "<think>" in content or in_think:
                                thinking_buffer += content
                                if "</think>" in thinking_buffer:
                                    # loại bỏ toàn bộ block think
                                    thinking_buffer = re.sub(r"<think>.*?</think>", "", thinking_buffer, flags=re.DOTALL)
                                    if thinking_buffer:
                                        yield thinking_buffer
                                    thinking_buffer = ""
                                    in_think = False
                                else:
                                    in_think = True
                                continue
                            # strip stray tags
                            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
                            content = content.replace("<think>", "").replace("</think>", "")
                            if not content:
                                continue
                        yield content
        except requests.exceptions.RequestException as e:
            yield f"\n[Lỗi kết nối Ollama LLM ({model_name}): {str(e)}]"
