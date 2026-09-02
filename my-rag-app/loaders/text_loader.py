from loaders.base import BaseLoader

class TextLoader(BaseLoader):
    """SRP: Đọc file văn bản thuần (.txt, .md)."""

    def load(self, file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
