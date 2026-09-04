import re
from typing import List
import requests
from config import config


def preprocess_query_text(text: str) -> str:
    """Trích nội dung trong "" nếu có, nếu không lấy toàn bộ để tránh loãng vector."""
    matches = re.findall(r'"([^"]+)"', text)
    if matches:
        return " ".join(matches).strip()
    return text.strip()


class OllamaEmbeddingService:
    """SRP: Gọi Ollama REST API để tạo Vector Embedding với query/document prefix (EmbeddingGemma)."""

    def __init__(self, host: str = config.OLLAMA_HOST, batch_size: int = 16):
        self.host = host
        self.api_url = f"{host}/api/embed"
        self.batch_size = batch_size
        self.query_prefix = config.EMBED_QUERY_PREFIX
        self.doc_prefix = config.EMBED_DOC_PREFIX

    def embed_query(self, text: str, model_name: str = config.EMBED_MODEL) -> List[float]:
        """Embed query với task prefix để phân biệt asymmetric encoding."""
        cleaned_text = preprocess_query_text(text)
        if not cleaned_text:
            cleaned_text = text.strip()
        prefixed = self.query_prefix + cleaned_text
        return self._embed_single(prefixed, model_name)

    def embed_documents(self, texts: List[str], model_name: str = config.EMBED_MODEL,
                        progress_callback=None) -> List[List[float]]:
        """Embed documents với task prefix + batch processing."""
        prefixed_texts = [self.doc_prefix + t for t in texts]
        return self.embed_batch(prefixed_texts, model_name, progress_callback)

    def embed_text(self, text: str, model_name: str = config.EMBED_MODEL) -> List[float]:
        """Legacy: embed without prefix (backward compat)."""
        return self._embed_single(text, model_name)

    def _embed_single(self, text: str, model_name: str) -> List[float]:
        """Embed 1 text (internal)."""
        embeddings = self.embed_batch([text], model_name=model_name)
        return embeddings[0]

    def embed_batch(self, texts: List[str], model_name: str = config.EMBED_MODEL,
                    progress_callback=None) -> List[List[float]]:
        """Tạo embedding cho danh sách văn bản theo batch để tránh timeout."""
        all_embeddings = []
        total_texts = len(texts)

        for i in range(0, total_texts, self.batch_size):
            batch_texts = texts[i: i + self.batch_size]
            payload = {
                "model": model_name,
                "input": batch_texts,
                "keep_alive": config.OLLAMA_KEEP_ALIVE
            }
            try:
                response = requests.post(self.api_url, json=payload, timeout=180)
                response.raise_for_status()
                data = response.json()
                all_embeddings.extend(data["embeddings"])

                if progress_callback:
                    progress_callback(min(i + self.batch_size, total_texts), total_texts)
            except requests.exceptions.RequestException as e:
                raise RuntimeError(
                    f"Lỗi kết nối Ollama Embedding ({model_name}) tại batch {i // self.batch_size + 1}: {str(e)}"
                )

        return all_embeddings
