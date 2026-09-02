from typing import List
import requests
from config import config

class OllamaEmbeddingService:
    """SRP: Gọi Ollama REST API để tạo Vector Embedding (Hỗ trợ model_name động)."""

    def __init__(self, host: str = config.OLLAMA_HOST):
        self.host = host
        self.api_url = f"{host}/api/embed"

    def embed_text(self, text: str, model_name: str = config.EMBED_MODEL) -> List[float]:
        """Tạo embedding cho 1 câu/đoạn văn bản."""
        embeddings = self.embed_batch([text], model_name=model_name)
        return embeddings[0]

    def embed_batch(self, texts: List[str], model_name: str = config.EMBED_MODEL) -> List[List[float]]:
        """Tạo embedding cho danh sách đoạn văn bản."""
        payload = {
            "model": model_name,
            "input": texts,
            "keep_alive": config.OLLAMA_KEEP_ALIVE
        }
        try:
            response = requests.post(self.api_url, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            return data["embeddings"]
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Lỗi kết nối Ollama Embedding ({model_name}): {str(e)}")
