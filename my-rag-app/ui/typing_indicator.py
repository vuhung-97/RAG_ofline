"""Typing indicator - 3 bouncing dots without text."""

from PyQt6.QtWidgets import QFrame, QHBoxLayout, QVBoxLayout, QLabel, QWidget
from PyQt6.QtCore import Qt, QTimer, QDateTime
from PyQt6.QtGui import QFont


class TypingIndicator(QFrame):
    """Bubble 3 chấm nhảy, ẩn khi token đầu về."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._frame = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._dots = []
        self._setup_ui()

    def _setup_ui(self):
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setStyleSheet("background: transparent;")

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(8, 2, 8, 2)
        main_layout.setSpacing(8)

        icon_label = QLabel("🤖")
        icon_label.setFixedSize(28, 28)
        icon_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        icon_label.setStyleSheet("font-size: 16pt; background: transparent;")

        content_widget = QWidget()
        content_widget.setStyleSheet("background: transparent;")
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(2)

        # Header: Trợ lý + time
        header = QHBoxLayout()
        header.setSpacing(8)
        name_label = QLabel("Trợ lý")
        nf = QFont()
        nf.setPointSize(9)
        nf.setBold(True)
        name_label.setFont(nf)
        name_label.setStyleSheet("color: #047857; background: transparent; border: none;")
        time_label = QLabel(QDateTime.currentDateTime().toString("HH:mm"))
        time_label.setStyleSheet("color: #94a3b8; font-size: 8pt; background: transparent; border: none;")
        header.addWidget(name_label)
        header.addWidget(time_label)
        header.addStretch()
        content_layout.addLayout(header)

        # Bubble chứa 3 chấm
        bubble = QWidget()
        bubble.setStyleSheet("""
            QWidget {
                background-color: #f0fdf4;
                border: 1px solid #bbf7d0;
                border-radius: 8px;
            }
        """)
        dots_layout = QHBoxLayout(bubble)
        dots_layout.setContentsMargins(12, 8, 12, 8)
        dots_layout.setSpacing(6)

        for _ in range(3):
            lbl = QLabel("●")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setFixedSize(14, 14)
            lbl.setStyleSheet("background: transparent; border: none; color: #047857; font-size: 8pt;")
            dots_layout.addWidget(lbl)
            self._dots.append(lbl)

        content_layout.addWidget(bubble)

        main_layout.addWidget(icon_label, 0, Qt.AlignmentFlag.AlignTop)
        main_layout.addWidget(content_widget)
        main_layout.addStretch()

    def start(self):
        self._frame = 0
        self._apply_frame()
        self._timer.start(250)

    def stop(self):
        self._timer.stop()

    def _tick(self):
        self._frame = (self._frame + 1) % 3
        self._apply_frame()

    def _apply_frame(self):
        # nhảy luân phiên: chấm active to + đậm, còn lại nhỏ + mờ
        for i, lbl in enumerate(self._dots):
            is_active = (i == self._frame)
            if is_active:
                lbl.setStyleSheet("background: transparent; border: none; color: #047857; font-size: 11pt;")
            else:
                lbl.setStyleSheet("background: transparent; border: none; color: #86efac; font-size: 7pt;")
