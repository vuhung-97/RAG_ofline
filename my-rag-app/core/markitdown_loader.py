"""MarkItDown Loader — thay thế tất cả loaders (DOCX, PDF, PPTX, XLSX, TXT).

Chuyển file → Markdown → chunks bằng MarkItDown + LangChain splitter.
"""

import re
from markitdown import MarkItDown
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter
)
from langchain_core.documents import Document
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

    def _preprocess_tables(self, markdown: str) -> str:
        """Detect Markdown tables và bọc bằng TABLE_START/END markers.

        - Lùi 1 dòng trước, tiếp 1 dòng sau → đưa vào nếu hợp lệ
        - Gán table_id tuần tự (1, 2, 3, ...)
        """
        lines = markdown.split("\n")
        result = []
        table_id = 0
        i = 0

        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Detect start of table: dòng bắt đầu bằng |
            if stripped.startswith("|"):
                table_id += 1

                # Lùi 1 dòng trước để lấy caption (nếu có)
                pre_line = None
                if result:
                    pre_candidate = result[-1].strip()
                    if pre_candidate and not pre_candidate.startswith("|"):
                        pre_line = result.pop()

                # Gom các dòng liên tiếp là bảng
                table_lines = []
                while i < len(lines) and lines[i].strip().startswith("|"):
                    table_lines.append(lines[i])
                    i += 1

                # Tiếp 1 dòng sau để lấy note (nếu có)
                post_line = None
                if i < len(lines):
                    post_candidate = lines[i].strip()
                    if post_candidate and not post_candidate.startswith("|"):
                        post_line = lines[i]
                        i += 1

                # Thêm TABLE_START marker
                result.append(f"<!-- TABLE_START_{table_id} -->")

                # Thêm caption (dòng trước bảng)
                if pre_line is not None:
                    result.append(pre_line)

                # Thêm nội dung bảng
                result.extend(table_lines)

                # Thêm note (dòng sau bảng)
                if post_line is not None:
                    result.append(post_line)

                # Thêm TABLE_END marker
                result.append(f"<!-- TABLE_END_{table_id} -->")
            else:
                result.append(lines[i])
                i += 1

        return "\n".join(result)

    def _set_table_metadata(self, doc: Document, table_ids_start: set, table_ids_end: set) -> Document:
        """Set metadata type cho chunk dựa trên table markers."""
        text = doc.page_content

        start_match = re.search(r'<!-- TABLE_START_(\d+) -->', text)
        end_match = re.search(r'<!-- TABLE_END_(\d+) -->', text)

        start_ids = set()
        end_ids = set()

        if start_match:
            start_ids = {int(m.group(1)) for m in re.finditer(r'<!-- TABLE_START_(\d+) -->', text)}
        if end_match:
            end_ids = {int(m.group(1)) for m in re.finditer(r'<!-- TABLE_END_(\d+) -->', text)}

        common = start_ids & end_ids

        if common:
            # Chunk chứa cả start và end của cùng 1 bảng → bảng hoàn chỉnh
            tid = common.pop()
            doc.metadata["type"] = "table"
            doc.metadata["table_id"] = tid
        elif start_ids:
            # Chỉ có start → table_start
            tid = start_ids.pop()
            doc.metadata["type"] = "table_start"
            doc.metadata["table_id"] = tid
        elif end_ids:
            # Chỉ có end → table_end
            tid = end_ids.pop()
            doc.metadata["type"] = "table_end"
            doc.metadata["table_id"] = tid
        else:
            # Kiểm tra xem có thuộc table nào không (dựa trên surrounding chunks)
            # Sẽ được xử lý sau khi split xong
            pass

        return doc

    def _assign_table_ids_to_middle_chunks(self, chunks: list) -> list:
        """Gán table_id cho các chunks ở giữa bảng (không có start/end marker).

        Dùng chunk_index để xác định chunks nào nằm giữa các table_start và table_end.
        """
        # Index các chunks theo table_id
        start_chunks = {}  # table_id → chunk
        end_chunks = {}    # table_id → chunk

        for chunk in chunks:
            meta = chunk.get("metadata", {})
            tid = meta.get("table_id")
            ctype = meta.get("type", "")
            if tid:
                if ctype == "table_start":
                    start_chunks[tid] = chunk
                elif ctype == "table_end":
                    end_chunks[tid] = chunk

        # Với mỗi table_id có start, tìm chunks nằm giữa
        for tid in set(start_chunks.keys()) | set(end_chunks.keys()):
            if tid not in start_chunks or tid not in end_chunks:
                continue

            start_idx = start_chunks[tid]["metadata"]["chunk_index"]
            end_idx = end_chunks[tid]["metadata"]["chunk_index"]

            for chunk in chunks:
                meta = chunk.get("metadata", {})
                if meta.get("table_id"):
                    continue  # Đã có table_id
                chunk_idx = meta.get("chunk_index", -1)
                if start_idx < chunk_idx < end_idx:
                    # Kiểm tra text có chứa | không (dòng bảng)
                    text = chunk.get("text", "")
                    has_table_row = any(
                        line.strip().startswith("|")
                        for line in text.split("\n")
                    )
                    if has_table_row:
                        meta["table_id"] = tid
                        meta["type"] = "table_middle"

        return chunks

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

        # Preprocess: detect tables, bọc markers
        markdown = self._preprocess_tables(markdown)

        # Split theo header (#, ##, ###)
        docs = self._md_splitter.split_text(markdown)

        # Split tiếp nếu section quá dài
        final_chunks = self._text_splitter.split_documents(docs)

        # Set metadata cho table chunks và xóa markers khỏi text
        for doc in final_chunks:
            self._set_table_metadata(doc, set(), set())
            # Xóa TABLE_START/END markers khỏi text (giữ nội dung)
            doc.page_content = re.sub(
                r'<!-- TABLE_START_\d+ -->\n?', '', doc.page_content
            )
            doc.page_content = re.sub(
                r'<!-- TABLE_END_\d+ -->\n?', '', doc.page_content
            )
            doc.page_content = doc.page_content.strip()

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

        # Gán table_id cho middle chunks
        chunks = self._assign_table_ids_to_middle_chunks(chunks)

        return chunks

    def is_supported(self, file_path: str) -> bool:
        """Kiểm tra file có hỗ trợ không."""
        return file_path.lower().endswith(self.SUPPORTED_EXTENSIONS)
