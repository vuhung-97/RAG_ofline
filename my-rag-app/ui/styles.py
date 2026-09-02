"""QSS Light Theme cho RAG Desktop App."""

LIGHT_THEME = """
/* ===== Global ===== */
QWidget {
    background-color: #f8fafc;
    color: #1e293b;
    font-family: "Segoe UI", "Roboto", sans-serif;
}

/* ===== Main Window ===== */
QMainWindow {
    background-color: #f8fafc;
}

/* ===== Sidebar ===== */
QWidget#sidebar {
    background-color: #ffffff;
    border-right: 1px solid #e2e8f0;
}

/* ===== Labels ===== */
QLabel {
    background: transparent;
    color: #334155;
}

QLabel#sidebar_title {
    font-size: 14pt;
    font-weight: 700;
    color: #1e293b;
    padding: 4px 0;
}

QLabel#section_header {
    font-size: 9pt;
    font-weight: 600;
    color: #64748b;
    padding: 8px 0 4px 0;
}

QLabel#file_item {
    font-size: 9pt;
    color: #475569;
    padding: 2px 4px;
}

/* ===== Buttons ===== */
QPushButton {
    background-color: #3b82f6;
    color: white;
    border: none;
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: 600;
    font-size: 9pt;
}

QPushButton:hover {
    background-color: #2563eb;
}

QPushButton:pressed {
    background-color: #1d4ed8;
}

QPushButton:disabled {
    background-color: #94a3b8;
    color: #e2e8f0;
}

QPushButton#btn_danger {
    background-color: #ef4444;
}

QPushButton#btn_danger:hover {
    background-color: #dc2626;
}

QPushButton#btn_secondary {
    background-color: #e2e8f0;
    color: #475569;
}

QPushButton#btn_secondary:hover {
    background-color: #cbd5e1;
}

QPushButton#send_button {
    background-color: #3b82f6;
    padding: 8px 20px;
    font-size: 10pt;
}

/* ===== ComboBox ===== */
QComboBox {
    background-color: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    padding: 6px 10px;
    min-height: 20px;
    color: #334155;
}

QComboBox:hover {
    border-color: #94a3b8;
}

QComboBox:focus {
    border-color: #3b82f6;
}

QComboBox::drop-down {
    border: none;
    width: 24px;
}

QComboBox::down-arrow {
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid #64748b;
    margin-right: 8px;
}

QComboBox QAbstractItemView {
    background-color: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    selection-background-color: #e0e7ff;
    selection-color: #1e293b;
    padding: 4px;
}

/* ===== Line Edit ===== */
QLineEdit {
    background-color: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    padding: 8px 10px;
    color: #1e293b;
    selection-background-color: #bfdbfe;
}

QLineEdit:focus {
    border-color: #3b82f6;
}

/* ===== Text Edit (Chat Input) ===== */
QTextEdit {
    background-color: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 8px;
    padding: 8px;
    color: #1e293b;
    selection-background-color: #bfdbfe;
}

QTextEdit:focus {
    border-color: #3b82f6;
}
QSpinBox, QDoubleSpinBox {
    background-color: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    padding: 6px 8px;
    color: #1e293b;
}

QSpinBox:focus, QDoubleSpinBox:focus {
    border-color: #3b82f6;
}

/* ===== Slider ===== */
QSlider::groove:horizontal {
    background: #e2e8f0;
    height: 6px;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #3b82f6;
    width: 16px;
    height: 16px;
    margin: -5px 0;
    border-radius: 8px;
}

QSlider::handle:horizontal:hover {
    background: #2563eb;
}

QSlider::sub-page:horizontal {
    background: #3b82f6;
    border-radius: 3px;
}

/* ===== Scroll Bar ===== */
QScrollBar:vertical {
    background: transparent;
    width: 8px;
    margin: 0;
}

QScrollBar::handle:vertical {
    background: #cbd5e1;
    border-radius: 4px;
    min-height: 30px;
}

QScrollBar::handle:vertical:hover {
    background: #94a3b8;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}

QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
    background: transparent;
}

QScrollBar:horizontal {
    background: transparent;
    height: 8px;
    margin: 0;
}

QScrollBar::handle:horizontal {
    background: #cbd5e1;
    border-radius: 4px;
    min-width: 30px;
}

QScrollBar::handle:horizontal:hover {
    background: #94a3b8;
}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0;
}

/* ===== Group Box ===== */
QGroupBox {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    margin-top: 12px;
    padding: 12px 8px 8px 8px;
    font-weight: 600;
    color: #334155;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 6px;
    background-color: #ffffff;
    color: #475569;
}

/* ===== Progress Bar ===== */
QProgressBar {
    background-color: #e2e8f0;
    border: none;
    border-radius: 4px;
    height: 8px;
    text-align: center;
    color: transparent;
}

QProgressBar::chunk {
    background-color: #3b82f6;
    border-radius: 4px;
}

/* ===== Splitter ===== */
QSplitter::handle {
    background-color: #e2e8f0;
    width: 2px;
}

QSplitter::handle:hover {
    background-color: #94a3b8;
}

/* ===== Chat Bubble Styles ===== */
QFrame#chat_bubble_user {
    background-color: transparent;
}

QFrame#chat_bubble_assistant {
    background-color: transparent;
}

QLabel#chat_timestamp {
    color: #94a3b8;
    font-size: 8pt;
}

/* ===== Dialog ===== */
QDialog {
    background-color: #f8fafc;
}

/* ===== Toast Notification ===== */
QLabel#toast {
    background-color: #1e293b;
    color: #f8fafc;
    border-radius: 8px;
    padding: 10px 20px;
    font-size: 9pt;
}

/* ===== Scroll Area ===== */
QScrollArea {
    background-color: transparent;
    border: none;
}

QScrollArea > QWidget > QWidget {
    background-color: transparent;
}
"""
