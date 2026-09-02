"""Chat message bubble widget - compact, markdown-like, collapsible sources."""

import re
from PyQt6.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QLabel, QWidget, QPushButton
)
from PyQt6.QtCore import Qt, QDateTime
from PyQt6.QtGui import QFont


class CollapsibleSources(QWidget):
    """Nguồn tham khảo có thể thu gọn/mở rộng bằng nút bấm."""

    def __init__(self, sources, parent=None):
        super().__init__(parent)
        self._is_expanded = False
        self._sources = sources

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(0)

        # Nút toggle
        count = len(sources[:3])
        self._toggle_btn = QPushButton(f"▸ 📌 Nguồn tham khảo ({count})")
        self._toggle_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                border: none;
                color: #475569;
                font-size: 8pt;
                font-weight: 600;
                text-align: left;
                padding: 4px 0;
            }
            QPushButton:hover {
                color: #1e40af;
            }
        """)
        self._toggle_btn.clicked.connect(self._toggle)
        layout.addWidget(self._toggle_btn)

        # Content container
        self._content_widget = QWidget()
        self._content_widget.setStyleSheet("background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 4px; padding: 4px;")
        content_layout = QVBoxLayout(self._content_widget)
        content_layout.setContentsMargins(8, 6, 8, 6)
        content_layout.setSpacing(6)

        for idx, src in enumerate(sources[:3], 1):
            fname = src.get("file_name", "Unknown")
            snippet = src.get("text", "")
            if len(snippet) > 150:
                snippet = snippet[:150] + "..."

            item_label = QLabel(f"[{idx}] {fname}")
            item_label.setStyleSheet("font-weight: 600; font-size: 8pt; color: #475569; background: transparent; border: none;")
            content_layout.addWidget(item_label)

            snippet_label = QLabel(snippet)
            snippet_label.setWordWrap(True)
            snippet_label.setStyleSheet("font-size: 8pt; color: #64748b; padding-left: 16px; background: transparent; border: none;")
            content_layout.addWidget(snippet_label)

        self._content_widget.setVisible(False)
        layout.addWidget(self._content_widget)

    def _toggle(self):
        self._is_expanded = not self._is_expanded
        self._content_widget.setVisible(self._is_expanded)
        arrow = "▾" if self._is_expanded else "▸"
        count = len(self._sources[:3])
        self._toggle_btn.setText(f"{arrow} 📌 Nguồn tham khảo ({count})")


class ChatMessageWidget(QFrame):
    """Widget hiển thị 1 tin nhắn chat - compact, markdown-like."""

    def __init__(self, role, text, sources=None, parent=None):
        super().__init__(parent)
        self.role = role
        self.text = text
        self.sources = sources or []
        self._text_label = None
        self._content_widget = None
        self._content_layout = None
        self._sources_added = False
        self._setup_ui()

    def _setup_ui(self):
        is_user = self.role == "user"
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setStyleSheet("background: transparent;")

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(8, 2, 8, 2)
        main_layout.setSpacing(8)

        # Icon
        icon_label = QLabel("👤" if is_user else "🤖")
        icon_label.setFixedSize(28, 28)
        icon_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        icon_label.setStyleSheet("font-size: 16pt; background: transparent;")

        # Content container - lưu thành instance để add_sources() dùng được
        self._content_widget = QWidget()
        self._content_widget.setStyleSheet("background: transparent;")
        self._content_layout = QVBoxLayout(self._content_widget)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(2)

        # Header: Tên + timestamp
        header_layout = QHBoxLayout()
        header_layout.setSpacing(8)

        timestamp = QDateTime.currentDateTime().toString("HH:mm")
        time_label = QLabel(timestamp)
        time_label.setStyleSheet("color: #94a3b8; font-size: 8pt; background: transparent; border: none;")

        name_label = QLabel("Bạn" if is_user else "Trợ lý")
        name_font = QFont()
        name_font.setPointSize(9)
        name_font.setBold(True)
        name_label.setFont(name_font)
        name_label.setStyleSheet(f"color: {'#1e40af' if is_user else '#047857'}; background: transparent; border: none;")

        # User: time | name (bên phải)
        # Assistant: name | time (bên trái)
        if is_user:
            header_layout.addStretch()
            header_layout.addWidget(time_label)
            header_layout.addWidget(name_label)
        else:
            header_layout.addWidget(name_label)
            header_layout.addWidget(time_label)
            header_layout.addStretch()

        self._content_layout.addLayout(header_layout)

        # Text content - QLabel compact, KHONG co font-size cứng
        self._text_label = QLabel()
        self._text_label.setWordWrap(True)
        self._text_label.setTextFormat(Qt.TextFormat.RichText)
        self._text_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        if is_user:
            self._text_label.setStyleSheet("""
                QLabel {
                    color: #1e293b;
                    background-color: #eff6ff;
                    border: 1px solid #bfdbfe;
                    border-radius: 8px;
                    padding: 6px 10px;
                }
            """)
        else:
            self._text_label.setStyleSheet("""
                QLabel {
                    color: #1e293b;
                    background-color: #f0fdf4;
                    border: 1px solid #bbf7d0;
                    border-radius: 8px;
                    padding: 6px 10px;
                }
            """)

        display_text = self._format_text(self.text)
        self._text_label.setText(display_text)
        self._content_layout.addWidget(self._text_label)
        # Apply initial font from QApplication if available
        try:
            from PyQt6.QtWidgets import QApplication
            app = QApplication.instance()
            if app:
                self._text_label.setFont(app.font())
        except Exception:
            pass

        # Sources - collapsible, chỉ cho assistant, nằm TRONG content (dưới text)
        if self.sources and not is_user:
            sources_widget = CollapsibleSources(self.sources)
            self._content_layout.addWidget(sources_widget)
            self._sources_added = True

        # Alignment: User = CẢ CỤM bên PHẢI, Assistant = CẢ CỤM bên TRÁI
        if is_user:
            main_layout.addStretch()
            main_layout.addWidget(self._content_widget)
            main_layout.addWidget(icon_label, 0, Qt.AlignmentFlag.AlignTop)
        else:
            main_layout.addWidget(icon_label, 0, Qt.AlignmentFlag.AlignTop)
            main_layout.addWidget(self._content_widget)
            main_layout.addStretch()

    def _format_text(self, text, font_size=None):
        """Format text thành HTML markdown-like. font_size inject để đồng bộ với app font."""
        if not text:
            return ""
        text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        # Bold **text**
        text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
        # Italic *text*
        text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
        # Code blocks ```...```
        text = re.sub(
            r'```(.*?)```',
            r'<pre style="background:#e2e8f0; padding:4px; border-radius:4px; font-family:Consolas,monospace;">\1</pre>',
            text, flags=re.DOTALL
        )
        # Inline code `...`
        text = re.sub(
            r'`(.*?)`',
            r'<code style="background:#e2e8f0; padding:1px 4px; border-radius:3px; font-family:Consolas,monospace;">\1</code>',
            text
        )
        # Line breaks
        text = text.replace("\n", "<br>")
        if font_size is not None:
            return f"<div style='line-height:1.5; font-size:{font_size}pt;'>{text}</div>"
        return f"<div style='line-height: 1.5;'>{text}</div>"

    def set_text(self, text, font_size=None):
        """Cập nhật text (dùng cho streaming)."""
        self.text = text
        if self._text_label:
            self._text_label.setText(self._format_text(text, font_size=font_size))

    def get_text_label(self):
        """Trả về QLabel để streaming append."""
        return self._text_label

    def add_sources(self, sources):
        """Thêm CollapsibleSources vào DƯỚI label text (trong QVBoxLayout dọc)."""
        if not sources or self.role == "user":
            return
        if self._sources_added:
            return
        if self._content_layout is None:
            return
        sources_widget = CollapsibleSources(sources)
        self._content_layout.addWidget(sources_widget)
        self._sources_added = True
        self.sources = sources

    def get_content_layout(self):
        """Trả về layout dọc chứa header + text + sources."""
        return self._content_layout
