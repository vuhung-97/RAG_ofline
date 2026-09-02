import os
from loaders.base import BaseLoader
from loaders.docx_loader import DocxLoader
from loaders.xlsx_loader import XlsxLoader
from loaders.pptx_loader import PptxLoader
from loaders.pdf_loader import PdfLoader
from loaders.text_loader import TextLoader

class LoaderFactory:
    """SRP & OCP: Quản lý mapping phần mở rộng file với Loader tương ứng."""

    def __init__(self):
        self._loaders = {
            ".docx": DocxLoader(),
            ".xlsx": XlsxLoader(),
            ".pptx": PptxLoader(),
            ".pdf": PdfLoader(),
            ".txt": TextLoader(),
            ".md": TextLoader()
        }

    def get_loader(self, file_path: str) -> BaseLoader:
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in self._loaders:
            raise ValueError(f"Định dạng file '{ext}' chưa được hỗ trợ.")
        return self._loaders[ext]

    def is_supported(self, file_path: str) -> bool:
        ext = os.path.splitext(file_path)[1].lower()
        return ext in self._loaders
