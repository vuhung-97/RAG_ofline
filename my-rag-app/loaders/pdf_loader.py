import pypdf
from loaders.base import BaseLoader

class PdfLoader(BaseLoader):
    """SRP: Trích xuất văn bản từ file PDF text-native."""

    def load(self, file_path: str) -> str:
        reader = pypdf.PdfReader(file_path)
        text_parts = []

        for page_idx, page in enumerate(reader.pages, 1):
            page_text = page.extract_text()
            if page_text and page_text.strip():
                text_parts.append(f"--- Page {page_idx} ---")
                text_parts.append(page_text.strip())

        return "\n".join(text_parts)
