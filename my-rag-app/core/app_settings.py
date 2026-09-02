"""Persist app settings to JSON (font, thinking, model params) - SRP: only load/save."""

import json
import os
from typing import Dict, Any
from config import config


SETTINGS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app_settings.json")

DEFAULTS: Dict[str, Any] = {
    "selected_llm": config.LLM_MODEL,
    "selected_embed": config.EMBED_MODEL,
    "num_ctx": config.LLM_NUM_CTX,
    "top_k": config.TOP_K,
    "temperature": config.TEMPERATURE,
    "chunk_size": config.CHUNK_SIZE,
    "chunk_overlap": config.CHUNK_OVERLAP,
    "font_size": config.FONT_SIZE,
    "enable_thinking": config.ENABLE_THINKING,
    "enable_rerank": config.ENABLE_RERANK,
}


class AppSettings:
    """Load/save settings JSON with merge defaults."""

    def __init__(self, file_path: str = SETTINGS_FILE):
        self.file_path = file_path

    def load(self) -> Dict[str, Any]:
        data = dict(DEFAULTS)
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                # only merge known keys
                for k in DEFAULTS:
                    if k in loaded:
                        data[k] = loaded[k]
            except Exception:
                pass
        return data

    def save(self, data: Dict[str, Any]) -> None:
        # merge with defaults to ensure all keys present
        merged = dict(DEFAULTS)
        merged.update({k: v for k, v in data.items() if k in DEFAULTS})
        # atomic write
        tmp = self.file_path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(merged, f, indent=2, ensure_ascii=False)
            os.replace(tmp, self.file_path)
        except Exception:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except Exception:
                    pass
            raise

    def get(self, key: str, default=None):
        data = self.load()
        return data.get(key, default)

    def set(self, key: str, value: Any):
        data = self.load()
        data[key] = value
        self.save(data)
