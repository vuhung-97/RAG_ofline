"""Chat input area với QTextEdit + Send/Stop button."""

from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QTextEdit, QPushButton, QLabel, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QFont, QKeyEvent


class ChatInputEdit(QTextEdit):
    """QTextEdit tùy chỉnh: Enter để gửi, Shift+Enter để xuống dòng."""

    submit_pressed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setPlaceholderText("Nhập câu hỏi của bạn...")
        self.setMaximumHeight(100)
        self.setMinimumHeight(40)

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if event.modifiers() == Qt.KeyboardModifier.NoModifier:
                self.submit_pressed.emit()
                return
        super().keyPressEvent(event)


class ChatInput(QWidget):
    """Input area cố định ở dưới cùng chat area."""

    message_submitted = pyqtSignal(str)
    stop_pressed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._is_stop_mode = False
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 8, 16, 12)
        main_layout.setSpacing(0)

        # Container frame
        container = QFrame()
        container.setFrameShape(QFrame.Shape.NoFrame)
        container.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 12px;
            }
        """)

        layout = QHBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Text input
        self.text_edit = ChatInputEdit()
        self.text_edit.setFrameShape(QFrame.Shape.NoFrame)
        self.text_edit.setStyleSheet("""
            QTextEdit {
                background-color: transparent;
                border: none;
                color: #1e293b;
                padding: 4px;
            }
        """)
        self.text_edit.submit_pressed.connect(self._on_submit)
        layout.addWidget(self.text_edit)

        # Send/Stop button
        self.send_button = QPushButton("Gửi")
        self.send_button.setObjectName("send_button")
        self.send_button.setFixedSize(QSize(60, 36))
        self.send_button.clicked.connect(self._on_submit)
        layout.addWidget(self.send_button, alignment=Qt.AlignmentFlag.AlignBottom)

        main_layout.addWidget(container)

    def _on_submit(self):
        if self._is_stop_mode:
            self.stop_pressed.emit()
            return
        text = self.text_edit.toPlainText().strip()
        if text:
            self.message_submitted.emit(text)
            self.text_edit.clear()

    def set_stop_mode(self):
        """Chuyển sang chế độ Dừng (đang stream)."""
        self._is_stop_mode = True
        self.send_button.setText("Dừng")
        self.send_button.setStyleSheet("""
            QPushButton {
                background-color: #ef4444;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: 600;
                font-size: 9pt;
            }
            QPushButton:hover {
                background-color: #dc2626;
            }
        """)

    def set_send_mode(self):
        """Quay lại chế độ Gửi."""
        self._is_stop_mode = False
        self.send_button.setText("Gửi")
        self.send_button.setStyleSheet("")
        self.send_button.setObjectName("send_button")
        # Re-apply send_button style from stylesheet
        self.send_button.style().unpolish(self.send_button)
        self.send_button.style().polish(self.send_button)

    def set_enabled(self, enabled):
        """Bật/tắt input."""
        self.text_edit.setReadOnly(not enabled)
        if not self._is_stop_mode:
            self.send_button.setEnabled(enabled)
        if enabled:
            self.text_edit.setFocus()

    def get_text(self):
        """Lấy text hiện tại."""
        return self.text_edit.toPlainText().strip()

    def update_font_size(self, font_size):
        """Cập nhật font size cho input và send button."""
        self.text_edit.setStyleSheet(f"""
            QTextEdit {{
                background-color: transparent;
                border: none;
                font-size: {font_size}pt;
                color: #1e293b;
                padding: 4px;
            }}
        """)
        font = self.send_button.font()
        font.setPointSize(font_size)
        self.send_button.setFont(font)
