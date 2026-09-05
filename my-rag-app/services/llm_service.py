from typing import List, Dict, Any, Generator
import json
import re
import time
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
        t_request_start = time.perf_counter()
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

            thinking_buffer = ""
            in_think = False
            first_token_received = False

            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line.decode("utf-8"))
                    if not enable_thinking and "message" in chunk and chunk["message"].get("thinking"):
                        continue
                    if "message" in chunk and "content" in chunk["message"]:
                        content = chunk["message"]["content"]
                        if not enable_thinking and content:
                            if "<think>" in content or in_think:
                                thinking_buffer += content
                                if "</think>" in thinking_buffer:
                                    thinking_buffer = re.sub(r"<think>.*?</think>", "", thinking_buffer, flags=re.DOTALL)
                                    if thinking_buffer:
                                        if not first_token_received:
                                            ttft = (time.perf_counter() - t_request_start) * 1000
                                            print(f"[LLM STREAM] ⚡ Token đầu tiên xuất hiện (TTFT): {ttft:.1f} ms")
                                            first_token_received = True
                                        yield thinking_buffer
                                    thinking_buffer = ""
                                    in_think = False
                                else:
                                    in_think = True
                                continue
                            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
                            content = content.replace("<think>", "").replace("</think>", "")
                            if not content:
                                continue

                        if not first_token_received and content:
                            ttft = (time.perf_counter() - t_request_start) * 1000
                            print(f"[LLM STREAM] ⚡ Token đầu tiên xuất hiện (TTFT): {ttft:.1f} ms")
                            first_token_received = True

                        yield content

            t_gen = (time.perf_counter() - t_request_start) * 1000
            print(f"[LLM STREAM] ✅ Sinh xong câu trả lời! [Thời gian sinh LLM: {t_gen:.1f} ms]")

        except requests.exceptions.RequestException as e:
            yield f"\n[Lỗi kết nối Ollama LLM ({model_name}): {str(e)}]"
