"""DocumentService — SRP: Điều phối luồng Ingest & Xóa Tài liệu."""

from typing import List, Dict, Any
from core.markitdown_loader import MarkItDownLoader
from core.chunk_processor import ChunkProcessor
from services.embedding_service import OllamaEmbeddingService
from core.vector_store import ChromaVectorStore
from core.bm25_index import BM25Index
from config import config


class DocumentService:
    """SRP & Orchestration: Điều phối luồng Ingest và quản lý Tài liệu."""

    def __init__(
        self,
        md_loader: MarkItDownLoader,
        embedding_service: OllamaEmbeddingService,
        vector_store: ChromaVectorStore,
        bm25_index: BM25Index = None
    ):
        self.md_loader = md_loader
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.bm25_index = bm25_index or BM25Index()

    def process_and_index_file(
        self,
        file_path: str,
        file_name: str,
        embed_model: str = config.EMBED_MODEL,
        progress_callback=None
    ) -> Dict[str, Any]:
        """Nạp file, cắt chunk, gán ID & links, embed với prefix và lưu vào Vector/BM25 DB."""
        indexed_files = self.vector_store.get_indexed_files()
        if file_name in indexed_files:
            return {"status": "skipped", "message": f"File '{file_name}' đã có trong nhóm '{self.vector_store.current_workspace}'."}

        # 1. Load + Chunk qua MarkItDown
        metadata = {"file_name": file_name, "file_path": file_path, "workspace": self.vector_store.current_workspace}
        chunks = self.md_loader.load_and_chunk(file_path, metadata)
        if not chunks:
            return {"status": "warning", "message": f"File '{file_name}' không chứa nội dung văn bản."}

        # 2. Xử lý gán Chunk ID (MD5) & liên kết prev/next IDs qua ChunkProcessor
        chunks = ChunkProcessor.process_and_link_chunks(chunks, file_name)

        # 3. Trích xuất texts và metadatas
        chunk_ids = [c["metadata"]["chunk_id"] for c in chunks]
        texts = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]

        # 4. Embed documents
        def embed_progress(current, total):
            if progress_callback:
                progress_callback(current, total)

        embeddings = self.embedding_service.embed_documents(
            texts, model_name=embed_model, progress_callback=embed_progress
        )

        # 5. Lưu vào ChromaDB Vector Store
        self.vector_store.add_documents(
            ids=chunk_ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )

        # 6. Lưu vào BM25 Index
        if config.BM25_ENABLED:
            self.bm25_index.add_documents(chunk_ids, texts, metadatas)
            try:
                self.bm25_index.save()
            except Exception:
                pass

        return {
            "status": "success",
            "message": f"Đã nạp thành công '{file_name}' ({len(chunks)} chunks) vào nhóm '{self.vector_store.current_workspace}'."
        }

    def remove_file(self, file_name: str) -> Dict[str, Any]:
        """Xóa file khỏi cả ChromaDB và BM25."""
        removed = self.vector_store.delete_file(file_name)

        if config.BM25_ENABLED and self.bm25_index.is_loaded:
            ids_to_remove = [
                cid for cid, meta in zip(
                    self.bm25_index.corpus_ids,
                    self.bm25_index.corpus_metadatas
                )
                if meta.get("file_name") == file_name
            ]
            if ids_to_remove:
                self.bm25_index.remove_documents(ids_to_remove)
                try:
                    self.bm25_index.save()
                except Exception:
                    pass

        return {
            "status": "success",
            "message": f"Đã xóa '{file_name}' ({removed} chunks)."
        }

    def rebuild_bm25_index(self):
        """Rebuild BM25 index từ dữ liệu ChromaDB hiện tại."""
        if not config.BM25_ENABLED:
            return

        try:
            all_data = self.vector_store.collection.get()
            if not all_data or not all_data["ids"]:
                self.bm25_index.clear()
                return

            ids = all_data["ids"]
            documents = all_data["documents"]
            metadatas = all_data["metadatas"]

            self.bm25_index.build(ids, documents, metadatas)
            self.bm25_index.save()
        except Exception:
            pass
