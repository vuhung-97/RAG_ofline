"""Settings dialog cho việc cấu hình model và tham số."""

import math
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel,
    QComboBox, QSlider, QSpinBox, QDoubleSpinBox, QPushButton,
    QGroupBox, QFrame, QCheckBox
)
from PyQt6.QtCore import Qt, pyqtSignal

OVERHEAD_TOKENS = 1000
MAX_CHUNK_TOKENS = 750  # CHUNK_SIZE = 750 chars, neighbors merged vào → chunk lớn hơn


class SettingsDialog(QDialog):
    """Dialog cài đặt model và tham số."""

    settings_saved = pyqtSignal(dict)

    def __init__(self, current_settings, installed_models, parent=None):
        super().__init__(parent)
        self.current_settings = current_settings
        self.installed_models = installed_models
        self.setWindowTitle("⚙️ Cài Đặt Model & Tham Số")
        self.setMinimumWidth(450)
        self.setMinimumHeight(500)
        self._setup_ui()
        self._load_current_values()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(12)

        # ===== Section 1: Model Selection =====
        model_group = QGroupBox("🤖 Chọn Model")
        model_layout = QFormLayout()
        model_layout.setSpacing(10)

        # LLM Model - hiện đủ model để bạn tự chọn vai trò
        all_models = self.installed_models.get("all", []) or self.installed_models.get("llm", []) + self.installed_models.get("embed", [])
        # dedup giữ thứ tự
        seen = set()
        uniq_all = []
        for m in all_models:
            if m not in seen:
                seen.add(m)
                uniq_all.append(m)
        self.llm_combo = QComboBox()
        self.llm_combo.addItems(uniq_all)
        self.llm_combo.setToolTip("Chọn bất kỳ model nào làm LLM - do bạn quyết định")
        model_layout.addRow("LLM Model:", self.llm_combo)

        # Embedding Model - hiện đủ model để bạn tự chọn vai trò
        self.embed_combo = QComboBox()
        self.embed_combo.addItems(uniq_all)
        self.embed_combo.setToolTip("Chọn bất kỳ model nào làm Embedding - do bạn quyết định")
        model_layout.addRow("Embedding Model:", self.embed_combo)

        model_group.setLayout(model_layout)
        main_layout.addWidget(model_group)

        # ===== Section 2: Parameters =====
        params_group = QGroupBox("🛠️ Tham Số Chi Tiết")
        params_layout = QFormLayout()
        params_layout.setSpacing(10)

        # Context Window (num_ctx)
        self.num_ctx_slider = QSlider(Qt.Orientation.Horizontal)
        self.num_ctx_slider.setMinimum(2048)
        self.num_ctx_slider.setMaximum(32768)
        self.num_ctx_slider.setSingleStep(2048)
        self.num_ctx_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.num_ctx_slider.setTickInterval(2048)
        self.num_ctx_label = QLabel("8192")
        self.num_ctx_slider.valueChanged.connect(
            lambda v: self.num_ctx_label.setText(str(v))
        )
        num_ctx_row = QHBoxLayout()
        num_ctx_row.addWidget(self.num_ctx_slider)
        num_ctx_row.addWidget(self.num_ctx_label)
        params_layout.addRow("Context Window (num_ctx):", num_ctx_row)

        # Top-K
        self.top_k_spin = QSpinBox()
        self.top_k_spin.setMinimum(1)
        self.top_k_spin.setMaximum(10)
        self.top_k_spin.setValue(4)
        self.top_k_spin.valueChanged.connect(self._on_top_k_changed)
        params_layout.addRow("Số đoạn tra cứu (Top-K):", self.top_k_spin)

        # Info label (num_ctx tự động)
        self.info_label = QLabel("")
        self.info_label.setStyleSheet("color: #64748b; font-size: 10px;")
        params_layout.addRow("💡 Tự động:", self.info_label)

        # Temperature
        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setMinimum(0.0)
        self.temp_spin.setMaximum(1.0)
        self.temp_spin.setSingleStep(0.1)
        self.temp_spin.setDecimals(1)
        self.temp_spin.setValue(0.7)
        params_layout.addRow("Độ sáng tạo (Temperature):", self.temp_spin)

        # Font Size
        self.font_spin = QSpinBox()
        self.font_spin.setMinimum(8)
        self.font_spin.setMaximum(20)
        self.font_spin.setSingleStep(1)
        self.font_spin.setValue(10)
        params_layout.addRow("Cỡ chữ (Font Size):", self.font_spin)

        # Thinking toggle
        self.thinking_check = QCheckBox("Bật Thinking (Qwen3)")
        self.thinking_check.setChecked(True)
        params_layout.addRow("Suy luận:", self.thinking_check)

        # Rerank toggle
        self.rerank_check = QCheckBox("Bật Rerank (dùng LLM đã chọn)")
        self.rerank_check.setChecked(True)
        params_layout.addRow("Rerank:", self.rerank_check)

        params_group.setLayout(params_layout)
        main_layout.addWidget(params_group)

        # ===== Separator =====
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color: #e2e8f0;")
        main_layout.addWidget(line)

        # ===== Buttons =====
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        cancel_btn = QPushButton("Hủy")
        cancel_btn.setObjectName("btn_secondary")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        save_btn = QPushButton("💾 Lưu Cài Đặt")
        save_btn.clicked.connect(self._on_save)
        btn_layout.addWidget(save_btn)

        main_layout.addLayout(btn_layout)

    @staticmethod
    def _next_power_of_2(n: int) -> int:
        """Trả về số mũ của 2 lớn hơn hoặc bằng n."""
        if n <= 0:
            return 2048
        return 1 << math.ceil(math.log2(n))

    def _on_top_k_changed(self, value: int):
        """Tự động tính num_ctx dựa trên top_k (power of 2)."""
        required = OVERHEAD_TOKENS + (value * MAX_CHUNK_TOKENS * 2)  # ×2 cho table merge
        num_ctx = self._next_power_of_2(required)
        num_ctx = max(2048, min(32768, num_ctx))

        self.num_ctx_slider.blockSignals(True)
        self.num_ctx_slider.setValue(num_ctx)
        self.num_ctx_label.setText(str(num_ctx))
        self.num_ctx_slider.blockSignals(False)

        self.info_label.setText(f"Tự động: {num_ctx} token (cho {value} chunks × {MAX_CHUNK_TOKENS})")

    def _load_current_values(self):
        """Load giá trị hiện tại vào form."""
        settings = self.current_settings

        # LLM Model
        llm = settings.get("selected_llm", "")
        idx = self.llm_combo.findText(llm)
        if idx >= 0:
            self.llm_combo.setCurrentIndex(idx)

        # Embed Model
        embed = settings.get("selected_embed", "")
        idx = self.embed_combo.findText(embed)
        if idx >= 0:
            self.embed_combo.setCurrentIndex(idx)

        # Parameters
        self.num_ctx_slider.setValue(settings.get("num_ctx", 8192))
        self.top_k_spin.setValue(settings.get("top_k", 6))
        self.temp_spin.setValue(settings.get("temperature", 0.3))
        self.font_spin.setValue(settings.get("font_size", 10))
        self.thinking_check.setChecked(settings.get("enable_thinking", True))
        self.rerank_check.setChecked(settings.get("enable_rerank", True))

        # Hiển thị info num_ctx tự động
        top_k_val = settings.get("top_k", 6)
        num_ctx_val = settings.get("num_ctx", 8192)
        self.info_label.setText(f"Tự động: {num_ctx_val} token (cho {top_k_val} chunks × {MAX_CHUNK_TOKENS})")

    def _on_save(self):
        """Lưu cài đặt và emit signal."""
        result = {
            "selected_llm": self.llm_combo.currentText(),
            "selected_embed": self.embed_combo.currentText(),
            "num_ctx": self.num_ctx_slider.value(),
            "top_k": self.top_k_spin.value(),
            "temperature": self.temp_spin.value(),
            "font_size": self.font_spin.value(),
            "enable_thinking": self.thinking_check.isChecked(),
            "enable_rerank": self.rerank_check.isChecked(),
        }
        self.settings_saved.emit(result)
        self.accept()

    def get_settings(self):
        """Lấy cài đặt đã chọn."""
        return {
            "selected_llm": self.llm_combo.currentText(),
            "selected_embed": self.embed_combo.currentText(),
            "num_ctx": self.num_ctx_slider.value(),
            "top_k": self.top_k_spin.value(),
            "temperature": self.temp_spin.value(),
            "font_size": self.font_spin.value(),
            "enable_thinking": self.thinking_check.isChecked(),
            "enable_rerank": self.rerank_check.isChecked(),
        }
