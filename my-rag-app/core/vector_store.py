import os
import re
import unicodedata
from typing import List, Dict, Any
import chromadb
from config import config

class ChromaVectorStore:
    """SRP: Quản lý Cơ sở dữ liệu Vector ChromaDB với hỗ trợ nhiều Workspace (Collection)."""

    def __init__(self, persist_dir: str = config.CHROMA_PERSIST_DIR):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.current_workspace = config.DEFAULT_WORKSPACE
        self.collection = self.client.get_or_create_collection(name=self._sanitize_name(self.current_workspace))

    def _sanitize_name(self, name: str) -> str:
        """Chuyển tên workspace thành tên collection ASCII hợp lệ trong ChromaDB (chỉ chứa a-z, 0-9, _, -)."""
        # Bỏ dấu tiếng Việt
        normalized = unicodedata.normalize('NFKD', name)
        ascii_text = ''.join([c for c in normalized if not unicodedata.combining(c)])
        # Thay ký tự đặc biệt bằng _
        clean_name = re.sub(r'[^a-zA-Z0-9_-]', '_', ascii_text).strip('_')
        if len(clean_name) < 3:
            clean_name = f"ws_{clean_name}"
        return clean_name[:63].lower()

    def set_workspace(self, workspace_name: str):
        """Đổi nhóm tài liệu / workspace hiện tại."""
        self.current_workspace = workspace_name
        sanitized = self._sanitize_name(workspace_name)
        self.collection = self.client.get_or_create_collection(name=sanitized)

    def list_workspaces(self) -> List[str]:
        """Lấy danh sách tên tất cả các workspace hiện có."""
        collections = self.client.list_collections()
        names = [c.name for c in collections]
        if not names:
            names = [self.current_workspace]
        return names

    def delete_workspace(self, workspace_name: str):
        """Xóa một workspace cụ thể - xóa luôn vật lý (folder UUID + VACUUM)."""
        sanitized = self._sanitize_name(workspace_name)
        # lấy uuid trước khi xóa để xóa folder vật lý
        uuid_to_delete = None
        try:
            cols = {c.name: getattr(c, "id", None) for c in self.client.list_collections()}
            uuid_to_delete = cols.get(sanitized)
        except Exception:
            pass
        try:
            self.client.delete_collection(name=sanitized)
        except Exception:
            pass
        # xóa folder vật lý chroma_db/<uuid> nếu còn
        if uuid_to_delete:
            try:
                import shutil
                folder = os.path.join(config.CHROMA_PERSIST_DIR, str(uuid_to_delete))
                if os.path.isdir(folder):
                    shutil.rmtree(folder)
            except Exception:
                pass
            # VACUUM để shrink sqlite (không bắt buộc, thử best-effort)
            try:
                import sqlite3
                db_path = os.path.join(config.CHROMA_PERSIST_DIR, "chroma.sqlite3")
                if os.path.exists(db_path):
                    con = sqlite3.connect(db_path)
                    con.execute("VACUUM")
                    con.close()
            except Exception:
                pass
        if self.current_workspace == workspace_name:
            self.set_workspace(config.DEFAULT_WORKSPACE)

    def delete_file(self, file_name: str) -> int:
        """Xóa 1 file trong workspace hiện tại - giữ nhóm, không xóa folder."""
        try:
            data = self.collection.get(where={"file_name": file_name})
            ids = data.get("ids", []) if data else []
            if ids:
                self.collection.delete(ids=ids)
                return len(ids)
        except Exception:
            pass
        return 0

    def add_documents(self, ids: List[str], embeddings: List[List[float]], documents: List[str], metadatas: List[Dict[str, Any]]):
        """Thêm danh sách vector và văn bản vào collection hiện tại."""
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

    def search_similarity(self, query_embedding: List[float], top_k: int = config.TOP_K) -> List[Dict[str, Any]]:
        """Tìm kiếm Top-K văn bản tương đồng trong collection hiện tại."""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        formatted_results = []
        if results and results["documents"] and len(results["documents"]) > 0:
            ids = results["ids"][0] if results["ids"] else []
            docs = results["documents"][0]
            metas = results["metadatas"][0] if results["metadatas"] else [{}] * len(docs)
            distances = results["distances"][0] if results["distances"] else [0.0] * len(docs)

            for cid, doc, meta, dist in zip(ids, docs, metas, distances):
                # Filter out poor quality results
                if dist > config.DISTANCE_THRESHOLD:
                    continue
                # Use chunk_id from metadata if available, else use ChromaDB ID
                if "chunk_id" not in meta:
                    meta["chunk_id"] = cid
                formatted_results.append({
                    "text": doc,
                    "metadata": meta,
                    "distance": dist
                })

        return formatted_results

    def get_indexed_files(self) -> List[str]:
        """Lấy danh sách các file đã được lưu trong workspace hiện tại."""
        all_data = self.collection.get()
        if not all_data or not all_data["metadatas"]:
            return []
        files = {meta.get("file_name") for meta in all_data["metadatas"] if meta.get("file_name")}
        return list(files)

    def clear_store(self):
        """Xóa toàn bộ dữ liệu trong workspace hiện tại - giữ tên nhóm (clear, không xóa vật lý tên)."""
        sanitized = self._sanitize_name(self.current_workspace)
        try:
            self.client.delete_collection(name=sanitized)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(name=sanitized)

    def clear_store_physical(self):
        """Xóa vật lý nhóm hiện tại - xóa luôn tên nhóm khỏi dropdown."""
        self.delete_workspace(self.current_workspace)
