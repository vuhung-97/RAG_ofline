"""RAG Desktop App - MainWindow entry point."""

import sys
import os

# Thêm root dir vào sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QSplitter, QWidget, QVBoxLayout, QLabel,
    QFrame, QHBoxLayout, QPushButton
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont

from config import config, get_installed_models
from loaders.factory import LoaderFactory
from core.text_splitter import TextSplitterService
from core.embedding_service import OllamaEmbeddingService
from core.vector_store import ChromaVectorStore
from core.llm_service import OllamaLLMService
from core.app_settings import AppSettings
from services.document_service import DocumentService
from services.rag_service import RAGService

from ui.sidebar import Sidebar
from ui.chat_area import ChatArea
from ui.chat_input import ChatInput
from ui.settings_dialog import SettingsDialog
from ui.workers import UploadWorker, StreamWorker, ModelLoaderWorker, FileProcessor
from ui.styles import LIGHT_THEME


class MainWindow(QMainWindow):
    """Main window của RAG Desktop App."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("RAG Tra Cuu Tai Lieu")
        self.setMinimumSize(900, 600)
        self.resize(1200, 750)

        # Load persisted settings (JSON) trước khi init
        self._app_settings = AppSettings()
        persisted = self._app_settings.load()
        self.font_size = persisted.get("font_size", config.FONT_SIZE)

        # Khởi tạo services
        self._init_services()

        # Session state (thay thế st.session_state) - load từ JSON
        self.messages = []
        self.selected_llm = persisted.get("selected_llm", config.LLM_MODEL)
        self.selected_embed = persisted.get("selected_embed", config.EMBED_MODEL)
        self.num_ctx = persisted.get("num_ctx", config.LLM_NUM_CTX)
        self.top_k = persisted.get("top_k", config.TOP_K)
        self.temperature = persisted.get("temperature", config.TEMPERATURE)
        self.chunk_size = persisted.get("chunk_size", config.CHUNK_SIZE)
        self.chunk_overlap = persisted.get("chunk_overlap", config.CHUNK_OVERLAP)
        self.enable_thinking = persisted.get("enable_thinking", config.ENABLE_THINKING)
        self.enable_rerank = persisted.get("enable_rerank", config.ENABLE_RERANK)

        # Workers
        self.upload_worker = None
        self.stream_worker = None
        self.model_worker = None

        # Temp files cần cleanup
        self._temp_files = []

        # Setup UI
        self._setup_ui()

        # Apply persisted font size ngay (không cần reset)
        self._apply_font_size(self.font_size)

        # Load models từ Ollama
        self._load_models()

    def _init_services(self):
        """Khởi tạo backend services."""
        self.loader_factory = LoaderFactory()
        self.splitter_service = TextSplitterService()
        self.embedding_service = OllamaEmbeddingService()
        self.vector_store = ChromaVectorStore()
        self.llm_service = OllamaLLMService()

        self.document_service = DocumentService(
            loader_factory=self.loader_factory,
            splitter_service=self.splitter_service,
            embedding_service=self.embedding_service,
            vector_store=self.vector_store
        )

        self.rag_service = RAGService(
            embedding_service=self.embedding_service,
            vector_store=self.vector_store,
            llm_service=self.llm_service
        )

    def _setup_ui(self):
        """Setup giao diện chính."""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ===== Header =====
        header = self._create_header()
        main_layout.addWidget(header)

        # ===== Content: Splitter =====
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(2)

        # Sidebar
        self.sidebar = Sidebar(self.vector_store)
        self.sidebar.setMinimumWidth(250)
        self.sidebar.setMaximumWidth(400)

        # Chat area + input
        chat_container = QWidget()
        chat_layout = QVBoxLayout(chat_container)
        chat_layout.setContentsMargins(0, 0, 0, 0)
        chat_layout.setSpacing(0)

        # Chat area
        self.chat_area = ChatArea()
        chat_layout.addWidget(self.chat_area, 1)

        # Chat input
        self.chat_input = ChatInput()
        chat_layout.addWidget(self.chat_input)

        # Thêm vào splitter
        splitter.addWidget(self.sidebar)
        splitter.addWidget(chat_container)
        splitter.setSizes([300, 900])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        main_layout.addWidget(splitter, 1)

        # ===== Status Bar =====
        self.statusBar().showMessage("Sẵn sàng")
        self.statusBar().setStyleSheet("color: #64748b; font-size: 9pt;")

        # ===== Connect signals =====
        self.sidebar.chat_cleared.connect(self._on_clear_chat)
        self.sidebar.workspace_changed.connect(self._on_workspace_changed)
        self.sidebar.workspace_created.connect(self._on_workspace_created)
        self.sidebar.files_uploaded.connect(self._on_files_uploaded)
        self.sidebar.settings_clicked.connect(self._on_settings_clicked)
        self.chat_input.message_submitted.connect(self._on_message_submitted)

        # Welcome message
        self.chat_area.add_message(
            "assistant",
            "Xin chào! Tôi là trợ lý tra cứu tài liệu RAG.\n\n"
            "Hãy nạp tài liệu vào sidebar và đặt câu hỏi để tôi tìm kiếm thông tin cho bạn."
        )

    def _create_header(self):
        """Tạo header bar."""
        header = QFrame()
        header.setFrameShape(QFrame.Shape.NoFrame)
        header.setFixedHeight(44)
        header.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border-bottom: 1px solid #e2e8f0;
            }
        """)

        layout = QHBoxLayout(header)
        layout.setContentsMargins(16, 6, 16, 6)

        # Workspace badge
        self.ws_badge = QLabel(config.DEFAULT_WORKSPACE)
        self.ws_badge.setStyleSheet("""
            QLabel {
                background-color: #e0e7ff;
                color: #3730a3;
                border: 1px solid #818cf8;
                border-radius: 10px;
                padding: 2px 12px;
                font-size: 9pt;
                font-weight: 600;
            }
        """)
        layout.addWidget(self.ws_badge)

        # Info label
        self.info_label = QLabel(
            f"LLM: {config.LLM_MODEL} | Context: {config.LLM_NUM_CTX} | Top-K: {config.TOP_K}"
        )
        self.info_label.setStyleSheet("color: #64748b; font-size: 9pt; background: transparent;")
        layout.addWidget(self.info_label)

        layout.addStretch()

        return header

    def _apply_font_size(self, font_size=None):
        """Áp dụng font size cho toàn bộ app - chỉ sau khi Lưu (không live)."""
        if font_size is None:
            font_size = self.font_size
        self.font_size = font_size
        font = QFont("Segoe UI", font_size)
        app = QApplication.instance()
        if app:
            app.setFont(font)
            # Không set lại LIGHT_THEME cứng (đã bỏ font-size khỏi styles.py)
            # Update tất cả chat messages hiện có với font_size inject vào HTML
            if hasattr(self, "chat_area"):
                self.chat_area._force_font_update(font_size)

    def _load_models(self):
        """Tải danh sách model từ Ollama."""
        self.model_worker = ModelLoaderWorker(config.OLLAMA_HOST)
        self.model_worker.models_ready.connect(self._on_models_loaded)
        self.model_worker.error.connect(lambda e: self.statusBar().showMessage(f"Không thể tải danh sách model: {e}"))
        self.model_worker.start()

    def _on_models_loaded(self, models):
        """Khi danh sách model được tải xong."""
        self.installed_models = models
        self.statusBar().showMessage("Đã tải danh sách model từ Ollama")

    def _on_clear_chat(self):
        """Xóa lịch sử chat."""
        self.messages = []
        self.chat_area.hide_typing()
        self.chat_area.clear_messages()
        self.chat_area.add_message(
            "assistant",
            "Đã xóa lịch sử chat. Hãy đặt câu hỏi mới!"
        )
        self.statusBar().showMessage("Đã xóa lịch sử chat")

    def _on_workspace_changed(self, name):
        """Khi đổi workspace."""
        self.ws_badge.setText(name)
        self.messages = []
        self.chat_area.clear_messages()
        self.chat_area.add_message(
            "assistant",
            f"Đã chuyển sang nhóm '{name}'. Hãy nạp tài liệu hoặc đặt câu hỏi!"
        )
        self.statusBar().showMessage(f"Đã chuyển sang nhóm: {name}")

    def _on_workspace_created(self, name):
        """Khi tạo workspace mới."""
        self.ws_badge.setText(name)
        self.messages = []
        self.chat_area.clear_messages()
        self.chat_area.add_message(
            "assistant",
            f"Đã tạo nhóm '{name}'. Hãy nạp tài liệu vào nhóm này!"
        )
        self.statusBar().showMessage(f"Đã tạo nhóm: {name}")

    def _on_files_uploaded(self, temp_paths, file_names):
        """Xử lý khi user upload file."""
        self._temp_files.extend(temp_paths)
        self.sidebar.show_progress(True, "Bắt đầu xử lý...")

        # Disable input trong khi upload
        self.chat_input.set_enabled(False)

        self.upload_worker = UploadWorker(
            self.document_service,
            temp_paths,
            file_names,
            self.selected_embed,
            self.chunk_size,
            self.chunk_overlap
        )
        self.upload_worker.progress.connect(self._on_upload_progress)
        self.upload_worker.finished.connect(self._on_upload_finished)
        self.upload_worker.error.connect(self._on_upload_error)
        self.upload_worker.start()

    def _on_upload_progress(self, percent, text):
        """Cập nhật tiến trình upload."""
        self.sidebar.update_progress(percent, text)

    def _on_upload_finished(self, result):
        """Khi upload file xong."""
        status = result.get("status", "")
        message = result.get("message", "")

        if status == "success":
            self.statusBar().showMessage(f"✅ {message}")
        elif status == "skipped":
            self.statusBar().showMessage(f"ℹ️ {message}")
        else:
            self.statusBar().showMessage(f"⚠️ {message}")

        self.sidebar.show_progress(False)
        self.sidebar.refresh_all()
        self.chat_input.set_enabled(True)

        # Cleanup temp files
        FileProcessor.cleanup_temp_files(self._temp_files)
        self._temp_files = []

    def _on_upload_error(self, error_msg):
        """Khi upload gặp lỗi."""
        self.sidebar.show_progress(False)
        self.chat_input.set_enabled(True)
        self.statusBar().showMessage(f"❌ Lỗi: {error_msg}")

        FileProcessor.cleanup_temp_files(self._temp_files)
        self._temp_files = []

    def _on_message_submitted(self, text):
        """Xử lý khi user gửi tin nhắn."""
        if not text.strip():
            return

        # Hiển thị tin nhắn user
        self.chat_area.add_message("user", text)
        self.messages.append({"role": "user", "content": text})

        # Disable input
        self.chat_input.set_enabled(False)
        self.statusBar().showMessage("Đang tìm kiếm & suy luận...")

        # Hiện 3 chấm nhảy khi chờ token đầu (không tạo bubble trống ngay)
        self.chat_area.show_typing()

        # Start stream worker
        self.stream_worker = StreamWorker(
            self.rag_service,
            text,
            self.messages[:-1],
            self.selected_llm,
            self.selected_embed,
            self.top_k,
            self.num_ctx,
            self.temperature,
            self.enable_thinking,
            self.enable_rerank
        )
        self.stream_worker.token_received.connect(self._on_token_received)
        self.stream_worker.sources_ready.connect(self._on_sources_ready)
        self.stream_worker.finished.connect(self._on_stream_finished)
        self.stream_worker.error.connect(self._on_stream_error)
        self.stream_worker.start()

    def _on_token_received(self, token):
        """Append token vào streaming widget - ẩn typing khi token đầu về."""
        if self.chat_area.is_typing_shown():
            self.chat_area.hide_typing()
            # tạo bubble thực ngay khi có token đầu
            if not self.chat_area._streaming_bubble:
                self.chat_area.create_streaming_message()
        self.chat_area.append_streaming_token(token)

    def _on_sources_ready(self, sources):
        """Lưu sources để dùng khi finish."""
        self._current_sources = sources

    def _on_stream_finished(self):
        """Khi stream xong."""
        # Nếu chưa có bubble (trường hợp không có token nào) thì ẩn typing và tạo bubble rỗng
        if self.chat_area.is_typing_shown():
            self.chat_area.hide_typing()
            if not self.chat_area._streaming_bubble:
                # không có token nhưng vẫn cần bubble để hiện sources hoặc rỗng
                self.chat_area.create_streaming_message()
        # Lấy full text từ streaming bubble
        if self.chat_area._streaming_bubble:
            full_text = self.chat_area._streaming_bubble.text
        else:
            full_text = ""

        sources = getattr(self, '_current_sources', [])

        # Finalize streaming
        self.chat_area.finalize_streaming(full_text, sources)

        # Lưu vào messages
        self.messages.append({
            "role": "assistant",
            "content": full_text,
            "sources": sources
        })

        # Enable input
        self.chat_input.set_enabled(True)
        self.statusBar().showMessage("Sẵn sàng")

    def _on_stream_error(self, error_msg):
        """Khi stream gặp lỗi."""
        if self.chat_area.is_typing_shown():
            self.chat_area.hide_typing()
            if not self.chat_area._streaming_bubble:
                self.chat_area.create_streaming_message()
        self.chat_area.finalize_streaming(f"❌ Lỗi: {error_msg}")
        self.chat_input.set_enabled(True)
        self.statusBar().showMessage(f"❌ Lỗi: {error_msg}")

    def _on_settings_clicked(self):
        """Mở dialog cài đặt."""
        current_settings = {
            "selected_llm": self.selected_llm,
            "selected_embed": self.selected_embed,
            "num_ctx": self.num_ctx,
            "top_k": self.top_k,
            "temperature": self.temperature,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "font_size": self.font_size,
            "enable_thinking": self.enable_thinking,
            "enable_rerank": self.enable_rerank,
        }

        installed = getattr(self, 'installed_models', {
            "llm": [config.LLM_MODEL],
            "embed": [config.EMBED_MODEL]
        })

        dialog = SettingsDialog(current_settings, installed, self)
        dialog.settings_saved.connect(self._on_settings_saved)
        dialog.exec()

    def _on_settings_saved(self, settings):
        """Lưu cài đặt mới - persist JSON và chỉ áp dụng font sau Lưu."""
        self.selected_llm = settings["selected_llm"]
        self.selected_embed = settings["selected_embed"]
        self.num_ctx = settings["num_ctx"]
        self.top_k = settings["top_k"]
        self.temperature = settings["temperature"]
        self.chunk_size = settings["chunk_size"]
        self.chunk_overlap = settings.get("chunk_overlap", self.chunk_overlap)
        self.font_size = settings.get("font_size", self.font_size)
        self.enable_thinking = settings.get("enable_thinking", self.enable_thinking)
        self.enable_rerank = settings.get("enable_rerank", self.enable_rerank)

        # Persist to JSON ngay
        try:
            self._app_settings.save({
                "selected_llm": self.selected_llm,
                "selected_embed": self.selected_embed,
                "num_ctx": self.num_ctx,
                "top_k": self.top_k,
                "temperature": self.temperature,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "font_size": self.font_size,
                "enable_thinking": self.enable_thinking,
                "enable_rerank": self.enable_rerank,
            })
        except Exception as e:
            self.statusBar().showMessage(f"⚠️ Lưu cài đặt lỗi: {e}")

        # Áp dụng font size sau khi Lưu
        self._apply_font_size(self.font_size)

        # Update info label
        self.info_label.setText(
            f"LLM: {self.selected_llm} | Context: {self.num_ctx} | Top-K: {self.top_k}"
        )
        self.statusBar().showMessage("Đã lưu cài đặt mới!")

    def closeEvent(self, event):
        """Cleanup khi đóng app."""
        # Cancel workers
        if self.upload_worker and self.upload_worker.isRunning():
            self.upload_worker.cancel()
            self.upload_worker.wait(2000)

        if self.stream_worker and self.stream_worker.isRunning():
            self.stream_worker.cancel()
            self.stream_worker.wait(2000)

        # Cleanup temp files
        FileProcessor.cleanup_temp_files(self._temp_files)

        event.accept()


def main():
    app = QApplication(sys.argv)

    # Apply stylesheet (đã bỏ font-size cứng)
    app.setStyleSheet(LIGHT_THEME)

    # Set font từ JSON persisted (fallback config)
    try:
        persisted = AppSettings().load()
        initial_font = persisted.get("font_size", config.FONT_SIZE)
    except Exception:
        initial_font = config.FONT_SIZE
    font = QFont("Segoe UI", initial_font)
    app.setFont(font)

    # Create and show main window
    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
