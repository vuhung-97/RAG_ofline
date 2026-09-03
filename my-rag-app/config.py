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
    TEMPERATURE: float = 0.0
    ENABLE_THINKING: bool = True

    # Embedding Prefixes (EmbeddingGemma asymmetric encoding)
    EMBED_QUERY_PREFIX: str = "task: search result | query: "
    EMBED_DOC_PREFIX: str = "task: search result | document: "
    EMBED_DIMENSION: int = 768

    # Chunking Configurations
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 150
    TABLE_MAX: int = 800
    CHUNK_TOKENS: int = 400
    OVERLAP_TOKENS: int = 150
    CHUNK_TOKENS_MAX: int = 500

    # UI Configurations
    FONT_SIZE: int = 10

    # Retrieval & Storage Configurations
    TOP_K: int = 6
    RERANK_CANDIDATES: int = 12
    ENABLE_RERANK: bool = True
    RERANK_TIMEOUT: int = 25
    DISTANCE_THRESHOLD: float = 1.10
    CHROMA_PERSIST_DIR: str = os.path.join(os.path.dirname(__file__), "chroma_db")
    DEFAULT_WORKSPACE: str = "Tài liệu chung"

    # Hybrid Search (BM25 + Semantic)
    BM25_ENABLED: bool = True
    BM25_TOP_K: int = 8
    BM25_INDEX_PATH: str = os.path.join(os.path.dirname(__file__), "bm25_index")
    SEMANTIC_TOP_K: int = 8
    FINAL_TOP_K: int = 8
    FUSION_METHOD: str = "rrf"
    RRF_K: int = 60

    # Similarity Threshold
    SIMILARITY_THRESHOLD: float = 0.75
    USE_ADAPTIVE_THRESHOLD: bool = True

    # Context Budget
    MAX_CONTEXT_CHUNKS: int = 8
    MAX_CONTEXT_TOKENS: int = 3000

    # Neighbor Expansion
    ENABLE_NEIGHBOR_EXPANSION: bool = True
    NEIGHBOR_SAME_SECTION_ONLY: bool = True
    NEIGHBOR_MAX_EXPANSION: int = 1

    # Performance Logging
    ENABLE_PERF_LOGGING: bool = True

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
