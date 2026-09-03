"""MarkItDown Loader — thay thế tất cả loaders (DOCX, PDF, PPTX, XLSX, TXT).

Chuyển file → Markdown → chunks bằng MarkItDown + LangChain splitter.
"""

from markitdown import MarkItDown
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter
)
from config import config


class MarkItDownLoader:
    """Load file bất kỳ → Markdown → chunks với header metadata."""

    SUPPORTED_EXTENSIONS = ('.docx', '.pdf', '.pptx', '.xlsx', '.txt', '.md', '.doc')

    def __init__(self):
        self._md = MarkItDown()
        self._headers = [
            ("#", "chapter"),
            ("##", "heading"),
            ("###", "subheading"),
        ]
        self._md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self._headers
        )
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def load_and_chunk(self, file_path: str, metadata: dict) -> list:
        """Convert file → Markdown → chunks.

        Args:
            file_path: Đường dẫn file.
            metadata: Metadata cơ bản (file_name, file_path, workspace).

        Returns:
            List[Dict] với keys "text" và "metadata".
        """
        result = self._md.convert(file_path)
        markdown = result.markdown

        if not markdown.strip():
            return []

        # Split theo header (#, ##, ###)
        docs = self._md_splitter.split_text(markdown)

        # Split tiếp nếu section quá dài
        final_chunks = self._text_splitter.split_documents(docs)

        # Convert về format hiện tại (List[Dict])
        chunks = []
        for i, doc in enumerate(final_chunks):
            chunk_meta = metadata.copy()
            chunk_meta.update(doc.metadata)
            chunk_meta["chunk_index"] = i
            chunks.append({
                "text": doc.page_content,
                "metadata": chunk_meta
            })

        return chunks

    def is_supported(self, file_path: str) -> bool:
        """Kiểm tra file có hỗ trợ không."""
        return file_path.lower().endswith(self.SUPPORTED_EXTENSIONS)
