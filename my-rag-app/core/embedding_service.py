from typing import List
import requests
from config import config

class OllamaEmbeddingService:
    """SRP: Gọi Ollama REST API để tạo Vector Embedding (Chia Batch để tránh Timeout file dài)."""

    def __init__(self, host: str = config.OLLAMA_HOST, batch_size: int = 16):
        self.host = host
        self.api_url = f"{host}/api/embed"
        self.batch_size = batch_size

    def embed_text(self, text: str, model_name: str = config.EMBED_MODEL) -> List[float]:
        """Tạo embedding cho 1 câu/đoạn văn bản."""
        embeddings = self.embed_batch([text], model_name=model_name)
        return embeddings[0]

    def embed_batch(self, texts: List[str], model_name: str = config.EMBED_MODEL, progress_callback=None) -> List[List[float]]:
        """Tạo embedding cho danh sách đoạn văn bản theo từng Batch nhỏ để không bị Read Timeout với file dài."""
        all_embeddings = []
        total_texts = len(texts)
        
        # Chia thành từng batch 16 chunks
        for i in range(0, total_texts, self.batch_size):
            batch_texts = texts[i : i + self.batch_size]
            payload = {
                "model": model_name,
                "input": batch_texts,
                "keep_alive": config.OLLAMA_KEEP_ALIVE
            }
            try:
                # Tăng timeout lên 180s cho mỗi batch
                response = requests.post(self.api_url, json=payload, timeout=180)
                response.raise_for_status()
                data = response.json()
                all_embeddings.extend(data["embeddings"])
                
                # Gọi callback cập nhật thanh tiến trình nếu có
                if progress_callback:
                    progress_callback(min(i + self.batch_size, total_texts), total_texts)
            except requests.exceptions.RequestException as e:
                raise RuntimeError(f"Lỗi kết nối Ollama Embedding ({model_name}) tại batch {i//self.batch_size + 1}: {str(e)}")

        return all_embeddings
