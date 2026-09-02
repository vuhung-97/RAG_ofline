import os
import requests
from dataclasses import dataclass

@dataclass
class Config:
    # Model Configurations
    EMBED_MODEL: str = "embeddinggemma:300m"
    LLM_MODEL: str = "qwen3:1.7b"
    OLLAMA_HOST: str = "http://localhost:11434"
    OLLAMA_KEEP_ALIVE: str = "5m"
    LLM_NUM_CTX: int = 8192
    TEMPERATURE: float = 0.7

    # Chunking Configurations
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    # Retrieval & Storage Configurations
    TOP_K: int = 4
    CHROMA_PERSIST_DIR: str = os.path.join(os.path.dirname(__file__), "chroma_db")
    DEFAULT_WORKSPACE: str = "Tài liệu chung"

config = Config()

def get_installed_models(host: str = config.OLLAMA_HOST) -> dict:
    """Lấy danh sách các model đã cài trong Ollama (tách LLM và Embedding)."""
    try:
        res = requests.get(f"{host}/api/tags", timeout=5)
        if res.status_code == 200:
            models = [m["name"] for m in res.json().get("models", [])]
            embed_models = [m for m in models if "embed" in m.lower() or "bge" in m.lower() or "gemma" in m.lower()]
            llm_models = [m for m in models if m not in embed_models or "qwen" in m.lower() or "gemma3" in m.lower()]
            return {
                "all": models,
                "embed": embed_models if embed_models else models,
                "llm": llm_models if llm_models else models
            }
    except Exception:
        pass
    return {"all": [config.LLM_MODEL, config.EMBED_MODEL], "embed": [config.EMBED_MODEL], "llm": [config.LLM_MODEL]}
