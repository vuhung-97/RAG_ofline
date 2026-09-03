"""Sidebar panel chứa workspace management, file upload, và settings."""

import os
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QLineEdit, QGroupBox, QFormLayout, QFileDialog, QProgressBar,
    QScrollArea, QFrame, QMessageBox, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QFont

from config import config


class Sidebar(QWidget):
    """Sidebar panel với controls cho workspace, upload, và settings."""

    # Signals
    chat_cleared = pyqtSignal()
    workspace_changed = pyqtSignal(str)
    workspace_created = pyqtSignal(str)
    files_uploaded = pyqtSignal(list, list)  # (temp_paths, file_names)
    settings_clicked = pyqtSignal()
    progress_update = pyqtSignal(int, str)

    def __init__(self, vector_store, parent=None):
        super().__init__(parent)
        self.vector_store = vector_store
        self.setObjectName("sidebar")
        self.setMinimumWidth(250)
        self.setMaximumWidth(400)
        self._setup_ui()
        self._refresh_workspaces()
        self._refresh_file_list()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(8)

        # Scroll area cho sidebar
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 4, 0)
        scroll_layout.setSpacing(12)

        # ===== Title =====
        title = QLabel("⚙️ Bảng Điều Khiển")
        title.setObjectName("sidebar_title")
        scroll_layout.addWidget(title)

        # ===== Section 1: Bộ Nhớ =====
        section1 = QLabel("1. BỘ NHỚ")
        section1.setObjectName("section_header")
        scroll_layout.addWidget(section1)

        btn_row = QHBoxLayout()
        self.btn_clear_chat = QPushButton("🗑️ Xóa Chat")
        self.btn_clear_chat.setObjectName("btn_secondary")
        self.btn_clear_chat.clicked.connect(self._on_clear_chat)
        btn_row.addWidget(self.btn_clear_chat)

        self.btn_clear_ws = QPushButton("⚠️ Xóa Nhóm")
        self.btn_clear_ws.setObjectName("btn_danger")
        self.btn_clear_ws.clicked.connect(self._on_clear_workspace)
        btn_row.addWidget(self.btn_clear_ws)

        scroll_layout.addLayout(btn_row)

        # Separator
        scroll_layout.addWidget(self._create_separator())

        # ===== Section 2: Nhóm Tài Liệu =====
        section2 = QLabel("2. NHÓM TÀI LIỆU & NẠP FILE")
        section2.setObjectName("section_header")
        scroll_layout.addWidget(section2)

        # Workspace selector
        self.workspace_combo = QComboBox()
        self.workspace_combo.currentTextChanged.connect(self._on_workspace_changed)
        scroll_layout.addWidget(self.workspace_combo)

        # Create workspace group
        create_group = QGroupBox("➕ Tạo Nhóm Tài Liệu Mới")
        create_layout = QHBoxLayout()
        self.new_ws_input = QLineEdit()
        self.new_ws_input.setPlaceholderText("Tên nhóm mới...")
        create_layout.addWidget(self.new_ws_input)

        self.btn_create_ws = QPushButton("Tạo")
        self.btn_create_ws.setFixedWidth(60)
        self.btn_create_ws.clicked.connect(self._on_create_workspace)
        create_layout.addWidget(self.btn_create_ws)

        create_group.setLayout(create_layout)
        scroll_layout.addWidget(create_group)

        # File list
        self.file_list_label = QLabel("📄 File đã nạp:")
        self.file_list_label.setObjectName("section_header")
        scroll_layout.addWidget(self.file_list_label)

        self.file_list_container = QWidget()
        self.file_list_layout = QVBoxLayout(self.file_list_container)
        self.file_list_layout.setContentsMargins(0, 0, 0, 0)
        self.file_list_layout.setSpacing(2)
        self.file_list_layout.addStretch()
        scroll_layout.addWidget(self.file_list_container)

        # Separator
        scroll_layout.addWidget(self._create_separator())

        # Upload section
        self.btn_upload = QPushButton("📥 Chọn File Để Nạp")
        self.btn_upload.clicked.connect(self._on_upload_files)
        scroll_layout.addWidget(self.btn_upload)

        # Progress bar (hidden by default)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(False)
        scroll_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("")
        self.status_label.setObjectName("file_item")
        self.status_label.setVisible(False)
        self.status_label.setWordWrap(True)
        scroll_layout.addWidget(self.status_label)

        # Separator
        scroll_layout.addWidget(self._create_separator())

        # Settings button
        self.btn_settings = QPushButton("⚙️ Cài Đặt Model")
        self.btn_settings.setObjectName("btn_secondary")
        self.btn_settings.clicked.connect(self.settings_clicked.emit)
        scroll_layout.addWidget(self.btn_settings)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        main_layout.addWidget(scroll)

    def _create_separator(self):
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color: #e2e8f0; max-height: 1px;")
        return line

    def _refresh_workspaces(self):
        """Làm mới danh sách workspace."""
        self.workspace_combo.blockSignals(True)
        self.workspace_combo.clear()

        workspaces = self.vector_store.list_workspaces()
        if config.DEFAULT_WORKSPACE not in workspaces:
            workspaces.append(config.DEFAULT_WORKSPACE)

        self.workspace_combo.addItems(workspaces)

        # Select current workspace
        current = self.vector_store.current_workspace
        idx = self.workspace_combo.findText(current)
        if idx >= 0:
            self.workspace_combo.setCurrentIndex(idx)

        self.workspace_combo.blockSignals(False)

    def _refresh_file_list(self):
        """Làm mới danh sách file đã nạp - mỗi dòng có nút xóa."""
        # Clear existing items
        while self.file_list_layout.count() > 1:
            item = self.file_list_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        files = self.vector_store.get_indexed_files()
        if files:
            self.file_list_label.setVisible(True)
            for f in files:
                row = QWidget()
                row_layout = QHBoxLayout(row)
                row_layout.setContentsMargins(0, 0, 0, 0)
                row_layout.setSpacing(4)

                label = QLabel(f"• {f}")
                label.setObjectName("file_item")
                label.setWordWrap(True)
                row_layout.addWidget(label, 1)

                btn_del = QPushButton("🗑️")
                btn_del.setFixedSize(28, 22)
                btn_del.setToolTip(f"Xóa file {f}")
                btn_del.setStyleSheet("""
                    QPushButton {
                        background-color: transparent;
                        border: 1px solid #e2e8f0;
                        border-radius: 4px;
                        font-size: 10pt;
                        padding: 0;
                    }
                    QPushButton:hover {
                        background-color: #fee2e2;
                        border-color: #fecaca;
                    }
                """)
                btn_del.clicked.connect(lambda _, fname=f: self._on_delete_file(fname))
                row_layout.addWidget(btn_del)

                self.file_list_layout.insertWidget(self.file_list_layout.count() - 1, row)
        else:
            self.file_list_label.setVisible(False)

    def _on_delete_file(self, file_name):
        """Xóa 1 file trong nhóm hiện tại - giữ nhóm, chỉ Yes/No."""
        reply = QMessageBox.question(
            self, "Xác nhận",
            f"Bạn có chắc muốn xóa file '{file_name}' khỏi nhóm '{self.vector_store.current_workspace}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            n = self.vector_store.delete_file(file_name)
            self._refresh_file_list()
            # chỉ xóa file, không xóa chat
            if n > 0:
                self.status_label.setText(f"Đã xóa {file_name} ({n} chunks)")
                self.status_label.setVisible(True)
                QTimer.singleShot(3000, lambda: self.status_label.setVisible(False))
            else:
                self.status_label.setText(f"Không tìm thấy file {file_name}")
                self.status_label.setVisible(True)

    def _on_workspace_changed(self, name):
        """Xử lý khi chọn workspace mới."""
        if name and name != self.vector_store.current_workspace:
            self.vector_store.set_workspace(name)
            self._refresh_file_list()
            self.workspace_changed.emit(name)

    def _on_create_workspace(self):
        """Tạo workspace mới."""
        name = self.new_ws_input.text().strip()
        if name:
            self.vector_store.set_workspace(name)
            self.new_ws_input.clear()
            self._refresh_workspaces()
            self._refresh_file_list()
            self.workspace_created.emit(name)

    def _on_clear_chat(self):
        """Xóa lịch sử chat."""
        reply = QMessageBox.question(
            self, "Xác nhận",
            "Bạn có chắc muốn xóa toàn bộ lịch sử chat?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.chat_cleared.emit()

    def _on_clear_workspace(self):
        """Xóa toàn bộ nhóm - xóa luôn tên nhóm vật lý."""
        ws_name = self.vector_store.current_workspace
        reply = QMessageBox.question(
            self, "Xác nhận",
            f"Bạn có chắc muốn xóa toàn bộ nhóm '{ws_name}'?\nTên nhóm sẽ bị xóa khỏi danh sách và không thể hoàn tác.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.vector_store.delete_workspace(ws_name)
            self._refresh_workspaces()
            self._refresh_file_list()
            self.workspace_changed.emit(self.vector_store.current_workspace)
            self.chat_cleared.emit()

    def _on_upload_files(self):
        """Mở file dialog để chọn file."""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Chọn file tài liệu",
            "",
            "Tất cả supported (*.docx *.xlsx *.pptx *.pdf *.txt *.md);;"
            "Word (*.docx);;"
            "Excel (*.xlsx);;"
            "PowerPoint (*.pptx);;"
            "PDF (*.pdf);;"
            "Text (*.txt *.md)"
        )

        if file_paths:
            from ui.workers import FileProcessor
            temp_paths, file_names = FileProcessor.save_uploaded_files(file_paths)
            self.files_uploaded.emit(temp_paths, file_names)

    def show_progress(self, visible=True, text=""):
        """Hiển thị/ẩn progress bar."""
        self.progress_bar.setVisible(visible)
        self.status_label.setVisible(visible and bool(text))
        self.status_label.setText(text)
        if not visible:
            self.progress_bar.setValue(0)

    def update_progress(self, percent, text=""):
        """Cập nhật progress bar."""
        self.progress_bar.setValue(percent)
        if text:
            self.status_label.setText(text)
            self.status_label.setVisible(True)

    def refresh_all(self):
        """Làm mới toàn bộ sidebar."""
        self._refresh_workspaces()
        self._refresh_file_list()
