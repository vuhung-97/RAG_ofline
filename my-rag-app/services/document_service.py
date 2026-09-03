import os
import hashlib
from typing import List, Dict, Any
from loaders.factory import LoaderFactory
from core.text_splitter import TextSplitterService
from core.embedding_service import OllamaEmbeddingService
from core.vector_store import ChromaVectorStore
from core.bm25_index import BM25Index
from config import config


class DocumentService:
    """SRP & Orchestration: Điều phối luồng Ingest Tài liệu."""

    def __init__(
        self,
        loader_factory: LoaderFactory,
        splitter_service: TextSplitterService,
        embedding_service: OllamaEmbeddingService,
        vector_store: ChromaVectorStore,
        bm25_index: BM25Index = None
    ):
        self.loader_factory = loader_factory
        self.splitter_service = splitter_service
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
        """Nạp file, cắt chunk, embed với prefix, lưu vào ChromaDB + BM25."""
        indexed_files = self.vector_store.get_indexed_files()
        if file_name in indexed_files:
            return {"status": "skipped", "message": f"File '{file_name}' đã có trong nhóm '{self.vector_store.current_workspace}'."}

        # 1. Load văn bản
        loader = self.loader_factory.get_loader(file_path)
        metadata = {"file_name": file_name, "file_path": file_path, "workspace": self.vector_store.current_workspace}
        chunks = None

        if hasattr(loader, "load_structured"):
            try:
                elements = loader.load_structured(file_path)
                if elements:
                    chunks = self.splitter_service.split_structured(elements, metadata)
            except Exception:
                chunks = None

        if chunks is None:
            raw_text = loader.load(file_path)
            if not raw_text.strip():
                return {"status": "warning", "message": f"File '{file_name}' không chứa nội dung văn bản."}
            self.splitter_service.splitter._chunk_size = config.CHUNK_SIZE
            self.splitter_service.splitter._chunk_overlap = config.CHUNK_OVERLAP
            chunks = self.splitter_service.split_text(raw_text, metadata)
        else:
            if not chunks:
                return {"status": "warning", "message": f"File '{file_name}' không chứa nội dung văn bản."}

        # 2. Gán chunk_id, prev_id, next_id
        chunk_ids = []
        for i, chunk in enumerate(chunks):
            chunk_id = hashlib.md5(
                f"{file_name}_{i}_{chunk['text'][:20]}".encode()
            ).hexdigest()
            chunk["metadata"]["chunk_id"] = chunk_id
            chunk_ids.append(chunk_id)

        # Link chunks
        for i in range(len(chunks)):
            if i > 0:
                chunks[i]["metadata"]["previous_chunk_id"] = chunk_ids[i - 1]
            if i < len(chunks) - 1:
                chunks[i]["metadata"]["next_chunk_id"] = chunk_ids[i + 1]

        # 3. Embed với document prefix
        texts = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]

        def embed_progress(current, total):
            if progress_callback:
                progress_callback(current, total)

        embeddings = self.embedding_service.embed_documents(
            texts, model_name=embed_model, progress_callback=embed_progress
        )

        # 4. Lưu vào ChromaDB
        self.vector_store.add_documents(
            ids=chunk_ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )

        # 5. Build BM25 index
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
