from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import config

class TextSplitterService:
    """SRP: Chia nhỏ đoạn văn bản thô thành các chunks có metadata."""

    def __init__(self, chunk_size: int = config.CHUNK_SIZE, chunk_overlap: int = config.CHUNK_OVERLAP):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def split_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Phân đoạn text và gán metadata cho từng chunk."""
        chunks = self.splitter.split_text(text)
        documents = []

        for idx, chunk in enumerate(chunks):
            doc_metadata = metadata.copy()
            doc_metadata["chunk_index"] = idx
            documents.append({
                "text": chunk,
                "metadata": doc_metadata
            })

        return documents
