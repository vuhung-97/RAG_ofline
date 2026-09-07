import os
import requests
from dataclasses import dataclass

@dataclass
class Config:
    # Model Configurations
    EMBED_MODEL: str = "embeddinggemma:300m"
    LLM_MODEL: str = "qwen3:4b-instruct"
    OLLAMA_HOST: str = "http://localhost:11434"
    OLLAMA_KEEP_ALIVE: str = "5m"
    LLM_NUM_CTX: int = 8192  # fallback mặc định — giá trị thực lấy từ app_settings.json
    TEMPERATURE: float = 0.1
    ENABLE_THINKING: bool = False  # instruct model = non-thinking

    # Embedding Prefixes (EmbeddingGemma asymmetric encoding)
    EMBED_QUERY_PREFIX: str = "task: search result | query: "
    EMBED_DOC_PREFIX: str = "task: search result | document: "
    EMBED_DOC_FORMAT: str = "title: {title} | text: {text}"  # Format mới cho document embedding
    EMBED_DIMENSION: int = 768
    EMBED_BATCH_SIZE: int = 16

    # Pipeline Tuning & Guardrail Thresholds
    CHAT_HISTORY_LIMIT: int = 4
    INTENT_BOOST_SCORE: float = 0.01
    GUARDRAIL_MIN_WORD_OVERLAP: float = 0.3

    # Chunking Configurations (MarkItDown + LangChain RecursiveCharacterTextSplitter)
    CHUNK_SIZE: int = 750
    CHUNK_OVERLAP: int = 200
    TABLE_MAX_CHARS: int = 8000  # Max ký tự cho 1 bảng sau merge (~2000 tokens)

    # UI Configurations
    FONT_SIZE: int = 10

    # Retrieval & Storage Configurations
    TOP_K: int = 6  # Giữ cho backward compat, mặc định dùng FINAL_TOP_K
    ENABLE_RERANK: bool = True
    DISTANCE_THRESHOLD: float = 1.4  # Cosine distance (0=identical, 1=orthogonal)
    CHROMA_PERSIST_DIR: str = os.path.join(os.path.dirname(__file__), "chroma_db")
    DEFAULT_WORKSPACE: str = "Tài liệu chung"

    # === Retrieval Pipeline (tách candidate vs final) ===
    SEMANTIC_TOP_K: int = 24       # Số chunk semantic retrieval lấy
    BM25_TOP_K: int = 24            # Số chunk BM25 retrieval lấy
    FUSION_TOP_K: int = 20          # Số chunk sau RRF fusion
    FINAL_TOP_K: int = 6            # Số evidence cuối cùng đưa vào context

    # === Index Versioning ===
    INDEX_SCHEMA_VERSION: int = 2
    EMBEDDING_PIPELINE_VERSION: int = 2

    # === Context Budget ===
    CONTEXT_BUDGET_EVIDENCE: int = 3500   # Token budget cho evidence
    CONTEXT_BUDGET_SYSTEM: int = 1000     # Token budget cho system + query + history
    CONTEXT_BUDGET_GENERATION: int = 3500 # Token reserve cho generation

    # === Table Expansion ===
    TABLE_EXPANSION_ENABLED: bool = True
    TABLE_MAX_CHUNKS: int = 30       # Giới hạn max chunks 1 bảng

    # === Debug Logging ===
    ENABLE_PIPELINE_DEBUG: bool = False  # Bật/tắt debug log từng bước

    # Hybrid Search (BM25 + Semantic)
    BM25_ENABLED: bool = True
    BM25_INDEX_PATH: str = os.path.join(os.path.dirname(__file__), "bm25_index")
    RRF_K: int = 60

    # Neighbor Expansion
    ENABLE_NEIGHBOR_EXPANSION: bool = True
    NEIGHBOR_SAME_SECTION_ONLY: bool = True
    NEIGHBOR_MAX_EXPANSION: int = 1

    # Performance Logging
    ENABLE_PERF_LOGGING: bool = True

    # Reranker
    RERANK_TIMEOUT: int = 60

config = Config()

def get_installed_models(host: str = config.OLLAMA_HOST) -> dict:
    """Lấy danh sách các model đã cài trong Ollama (tách LLM và Embedding)."""
    try:
        res = requests.get(f"{host}/api/tags", timeout=5)
        if res.status_code == 200:
            models = [m["name"] for m in res.json().get("models", [])]
            embed_models = [m for m in models if "embed" in m.lower()]
            llm_models = [m for m in models if m not in embed_models]
            return {
                "all": models,
                "embed": embed_models if embed_models else models,
                "llm": llm_models if llm_models else models
            }
    except Exception:
        pass
    return {"all": [config.LLM_MODEL, config.EMBED_MODEL], "embed": [config.EMBED_MODEL], "llm": [config.LLM_MODEL]}
