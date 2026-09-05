"""ChunkProcessor — SRP: gán ID duy nhất và liên kết prev/next IDs cho các chunk."""

import hashlib
from typing import List, Dict, Any


class ChunkProcessor:
    """Tạo chunk_id (MD5) và thiết lập liên kết previous_chunk_id / next_chunk_id."""

    @staticmethod
    def process_and_link_chunks(chunks: List[Dict[str, Any]], file_name: str) -> List[Dict[str, Any]]:
        """Gán chunk_id và tạo liên kết lân cận giữa các chunks."""
        if not chunks:
            return []

        # 1. Gán chunk_id
        chunk_ids = []
        for i, chunk in enumerate(chunks):
            chunk_id = hashlib.md5(
                f"{file_name}_{i}_{chunk['text'][:20]}".encode("utf-8")
            ).hexdigest()
            chunk["metadata"]["chunk_id"] = chunk_id
            chunk_ids.append(chunk_id)

        # 2. Tạo liên kết previous_chunk_id và next_chunk_id
        total = len(chunks)
        for i in range(total):
            if i > 0:
                chunks[i]["metadata"]["previous_chunk_id"] = chunk_ids[i - 1]
            if i < total - 1:
                chunks[i]["metadata"]["next_chunk_id"] = chunk_ids[i + 1]

        return chunks
