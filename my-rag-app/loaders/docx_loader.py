import docx
from loaders.base import BaseLoader

class DocxLoader(BaseLoader):
    """SRP: Trích xuất văn bản thô từ file Microsoft Word (.docx)."""

    def load(self, file_path: str) -> str:
        doc = docx.Document(file_path)
        text_parts = []

        # Đọc văn bản từ các paragraph
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text_parts.append(paragraph.text.strip())

        # Đọc văn bản từ các bảng (Tables)
        for table in doc.tables:
            for row in table.rows:
                row_text = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if row_text:
                    text_parts.append(" | ".join(row_text))

        return "\n".join(text_parts)
