from typing import List, Dict, Any
import chromadb
from config import config

class ChromaVectorStore:
    """SRP: Quản lý Cơ sở dữ liệu Vector ChromaDB (Index, Search, Persist)."""

    def __init__(self, persist_dir: str = config.CHROMA_PERSIST_DIR, collection_name: str = config.COLLECTION_NAME):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_documents(self, ids: List[str], embeddings: List[List[float]], documents: List[str], metadatas: List[Dict[str, Any]]):
        """Thêm danh sách vector và văn bản vào ChromaDB."""
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

    def search_similarity(self, query_embedding: List[float], top_k: int = config.TOP_K) -> List[Dict[str, Any]]:
        """Tìm kiếm Top-K văn bản tương đồng với query vector."""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        formatted_results = []
        if results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if results["metadatas"] else [{}] * len(docs)
            distances = results["distances"][0] if results["distances"] else [0.0] * len(docs)

            for doc, meta, dist in zip(docs, metas, distances):
                formatted_results.append({
                    "text": doc,
                    "metadata": meta,
                    "distance": dist
                })

        return formatted_results

    def get_indexed_files(self) -> List[str]:
        """Lấy danh sách các file đã được lưu trong DB."""
        all_data = self.collection.get()
        if not all_data or not all_data["metadatas"]:
            return []
        files = {meta.get("file_name") for meta in all_data["metadatas"] if meta.get("file_name")}
        return list(files)

    def clear_store(self):
        """Xóa toàn bộ dữ liệu trong collection."""
        self.client.delete_collection(name=config.COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(name=config.COLLECTION_NAME)
