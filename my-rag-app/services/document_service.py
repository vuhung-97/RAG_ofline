import os
import hashlib
from typing import List, Dict, Any
from loaders.factory import LoaderFactory
from core.text_splitter import TextSplitterService
from core.embedding_service import OllamaEmbeddingService
from core.vector_store import ChromaVectorStore

class DocumentService:
    """SRP & Orchestration: Điều phối luồng Ingest Tài liệu (Loader -> Splitter -> Embed -> Store)."""

    def __init__(
        self,
        loader_factory: LoaderFactory,
        splitter_service: TextSplitterService,
        embedding_service: OllamaEmbeddingService,
        vector_store: ChromaVectorStore
    ):
        self.loader_factory = loader_factory
        self.splitter_service = splitter_service
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def process_and_index_file(self, file_path: str, file_name: str) -> Dict[str, Any]:
        """Nạp file, cắt nhỏ, nhúng vector và lưu vào ChromaDB."""
        # Check xem file đã nạp chưa
        indexed_files = self.vector_store.get_indexed_files()
        if file_name in indexed_files:
            return {"status": "skipped", "message": f"File '{file_name}' đã có trong hệ thống."}

        # 1. Load văn bản
        loader = self.loader_factory.get_loader(file_path)
        raw_text = loader.load(file_path)

        if not raw_text.strip():
            return {"status": "warning", "message": f"File '{file_name}' không chứa nội dung văn bản."}

        # 2. Split chunks
        metadata = {"file_name": file_name, "file_path": file_path}
        chunks = self.splitter_service.split_text(raw_text, metadata)

        # 3. Tạo Embeddings & IDs
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        embeddings = self.embedding_service.embed_batch(texts)

        ids = [
            hashlib.md5(f"{file_name}_{idx}_{chunk['text'][:20]}".encode()).hexdigest()
            for idx, chunk in enumerate(chunks)
        ]

        # 4. Lưu vào Vector Store
        self.vector_store.add_documents(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )

        return {
            "status": "success",
            "message": f"Đã nạp thành công '{file_name}' ({len(chunks)} chunks)."
        }
