"""Chat area scrollable chứa các tin nhắn - compact streaming."""

import re
from PyQt6.QtWidgets import (
    QScrollArea, QWidget, QVBoxLayout, QLabel, QFrame
)
from PyQt6.QtCore import Qt, QTimer

from ui.chat_message import ChatMessageWidget


class ChatArea(QScrollArea):
    """Scrollable chat area hiển thị lịch sử tin nhắn."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._streaming_widget = None
        self._streaming_bubble = None
        self._typing_indicator = None

    def _setup_ui(self):
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setFrameShape(QFrame.Shape.NoFrame)

        self.container = QWidget()
        self.container.setStyleSheet("background-color: #f8fafc;")
        self.layout = QVBoxLayout(self.container)
        self.layout.setContentsMargins(16, 8, 16, 8)
        self.layout.setSpacing(4)
        self.layout.addStretch()

        self.setWidget(self.container)

    def add_message(self, role, text, sources=None):
        """Thêm tin nhắn vào chat area."""
        bubble = ChatMessageWidget(role, text, sources)
        self.layout.insertWidget(self.layout.count() - 1, bubble)
        QTimer.singleShot(50, self._scroll_to_bottom)
        return bubble

    def create_streaming_message(self):
        """Tạo bubble trống để stream token vào."""
        bubble = ChatMessageWidget("assistant", "")
        self.layout.insertWidget(self.layout.count() - 1, bubble)
        self._streaming_bubble = bubble
        self._streaming_widget = bubble.get_text_label()
        QTimer.singleShot(50, self._scroll_to_bottom)
        return bubble

    def append_streaming_token(self, token):
        """Append token vào streaming widget."""
        if self._streaming_widget and self._streaming_bubble:
            current = self._streaming_bubble.text
            new_text = current + token
            self._streaming_bubble.text = new_text
            # inherit current app font size for streaming
            try:
                from PyQt6.QtWidgets import QApplication
                fs = QApplication.instance().font().pointSize()
            except Exception:
                fs = None
            self._streaming_widget.setText(
                self._streaming_bubble._format_text(new_text, font_size=fs)
            )
            self._scroll_to_bottom()

    def replace_streaming_text(self, new_text):
        """Replace toàn bộ nội dung streaming bubble (dùng khi guardrail fail)."""
        if self._streaming_bubble:
            self._streaming_bubble.text = new_text
            try:
                from PyQt6.QtWidgets import QApplication
                fs = QApplication.instance().font().pointSize()
            except Exception:
                fs = None
            self._streaming_bubble._text_label.setText(
                self._streaming_bubble._format_text(new_text, font_size=fs)
            )
            self._scroll_to_bottom()

    def finalize_streaming(self, full_text, sources=None):
        """Hoàn thành streaming, set lại nội dung với formatting."""
        if self._streaming_bubble:
            self._streaming_bubble.text = full_text
            try:
                from PyQt6.QtWidgets import QApplication
                fs = QApplication.instance().font().pointSize()
            except Exception:
                fs = None
            self._streaming_bubble._text_label.setText(
                self._streaming_bubble._format_text(full_text, font_size=fs)
            )

            self._streaming_widget = None
            self._streaming_bubble = None

    def show_typing(self):
        """Hiện 3 chấm nhảy khi chờ token đầu."""
        if self._typing_indicator is not None:
            return
        from ui.typing_indicator import TypingIndicator
        self._typing_indicator = TypingIndicator()
        self.layout.insertWidget(self.layout.count() - 1, self._typing_indicator)
        self._typing_indicator.start()
        QTimer.singleShot(50, self._scroll_to_bottom)

    def hide_typing(self):
        """Ẩn 3 chấm nhảy."""
        if self._typing_indicator is None:
            return
        try:
            self._typing_indicator.stop()
        except Exception:
            pass
        self.layout.removeWidget(self._typing_indicator)
        self._typing_indicator.deleteLater()
        self._typing_indicator = None

    def is_typing_shown(self) -> bool:
        return self._typing_indicator is not None

    def clear_messages(self):
        """Xóa tất cả tin nhắn."""
        self.hide_typing()
        while self.layout.count() > 1:
            item = self.layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        self._streaming_widget = None
        self._streaming_bubble = None

    def _scroll_to_bottom(self):
        """Scroll xuống cuối cùng."""
        scrollbar = self.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _force_font_update(self, font_size=None):
        """Force cập nhật font cho tất cả tin nhắn hiện có."""
        from PyQt6.QtGui import QFont
        from PyQt6.QtWidgets import QApplication
        app = QApplication.instance()
        app_font = app.font()
        fs = font_size if font_size is not None else app_font.pointSize()
        for i in range(self.layout.count()):
            item = self.layout.itemAt(i)
            if item and item.widget():
                widget = item.widget()
                if isinstance(widget, ChatMessageWidget):
                    # Update text label font
                    label = widget.get_text_label()
                    if label:
                        label.setFont(app_font)
                    # Re-render text với font mới (inject font-size vào HTML)
                    if widget.text is not None:
                        label.setText(widget._format_text(widget.text, font_size=fs))
