import os
from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    # Model Configurations
    EMBED_MODEL: str = "embeddinggemma:300m"
    LLM_MODEL: str = "qwen3:1.7b"
    OLLAMA_HOST: str = "http://localhost:11434"
    OLLAMA_KEEP_ALIVE: str = "5m"
    LLM_NUM_CTX: int = 8192

    # Chunking Configurations
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    # Retrieval & Storage Configurations
    TOP_K: int = 4
    CHROMA_PERSIST_DIR: str = os.path.join(os.path.dirname(__file__), "chroma_db")
    COLLECTION_NAME: str = "rag_documents"

config = Config()
