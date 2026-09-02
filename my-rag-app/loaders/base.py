from abc import ABC, abstractmethod

class BaseLoader(ABC):
    """Abstract Base Class cho tất cả các Document Loaders (SRP Interface)."""
    
    @abstractmethod
    def load(self, file_path: str) -> str:
        """Đọc file và trả về toàn bộ nội dung văn bản thô."""
        pass
