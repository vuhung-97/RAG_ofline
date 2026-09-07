"""BM25 Index using bm25s library — SRP: chỉ index và search keyword.

Sửa: persistence fix (corpus_texts restore từ JSON), workspace isolation, invariant check.
bm25s 0.3.x: load() KHÔNG dùng load_corpus=True (bug), chỉ load index.
"""

import os
import re
import json
import unicodedata
from typing import List, Dict, Any, Tuple

import bm25s

from config import config


class BM25Index:
    """BM25 keyword search index with persistence và workspace isolation."""

    def __init__(self, index_path: str = config.BM25_INDEX_PATH, workspace: str = None):
        if workspace:
            self.index_path = os.path.join(index_path, self._sanitize_workspace(workspace))
        else:
            self.index_path = index_path
        self.corpus_ids: List[str] = []
        self.corpus_texts: List[str] = []
        self.corpus_metadatas: List[Dict] = []
        self.bm25 = None
        self._loaded = False

    @staticmethod
    def _sanitize_workspace(name: str) -> str:
        """Tạo tên folder an toàn từ workspace name."""
        normalized = unicodedata.normalize('NFKD', name)
        ascii_text = ''.join([c for c in normalized if not unicodedata.combining(c)])
        clean = re.sub(r'[^a-zA-Z0-9_-]', '_', ascii_text).strip('_')
        return clean[:50].lower() or "default"

    def _validate_invariant(self):
        """Đảm bảo len(ids) == len(texts) == len(metas)."""
        n = len(self.corpus_ids)
        assert n == len(self.corpus_texts), f"IDs({n}) != Texts({len(self.corpus_texts)})"
        assert n == len(self.corpus_metadatas), f"IDs({n}) != Metas({len(self.corpus_metadatas)})"

    def build(self, ids: List[str], texts: List[str], metadatas: List[Dict]):
        """Build BM25 index từ corpus."""
        self.corpus_ids = list(ids)
        self.corpus_texts = list(texts)
        self.corpus_metadatas = list(metadatas)

        # Tokenize manually - bm25s.tokenize expects list of strings
        tokenized_corpus = [self._tokenize(t) for t in texts]

        self.bm25 = bm25s.BM25()
        self.bm25.index(tokenized_corpus)
        self._loaded = True
        self._validate_invariant()

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float, Dict]]:
        """Search BM25, return [(chunk_id, score, metadata), ...]."""
        if not self._loaded or self.bm25 is None:
            return []

        # Clamp top_k to corpus size
        effective_k = min(top_k, len(self.corpus_ids))
        if effective_k == 0:
            return []

        tokenized_query = [self._tokenize(query)]
        results = self.bm25.retrieve(tokenized_query, k=effective_k, show_progress=False)

        output = []
        if results is not None:
            # bm25s 0.3.x uses 'documents' attribute for indices
            doc_indices = results.documents[0] if hasattr(results, 'documents') else []
            scores_arr = results.scores[0] if hasattr(results, 'scores') else []
            for idx_val, score_val in zip(doc_indices, scores_arr):
                idx = int(idx_val)
                if 0 <= idx < len(self.corpus_ids):
                    output.append((
                        self.corpus_ids[idx],
                        float(score_val),
                        self.corpus_metadatas[idx]
                    ))
        return output

    def add_documents(self, ids: List[str], texts: List[str], metadatas: List[Dict]):
        """Thêm documents vào index (rebuild toàn bộ)."""
        self.load()  # reload từ disk để đảm bảo corpus_ids là mới nhất
        all_ids = self.corpus_ids + list(ids)
        all_texts = self.corpus_texts + list(texts)
        all_metadatas = self.corpus_metadatas + list(metadatas)
        self.build(all_ids, all_texts, all_metadatas)

    def remove_documents(self, ids_to_remove: List[str]):
        """Xóa documents theo IDs (rebuild)."""
        remove_set = set(ids_to_remove)
        new_ids = []
        new_texts = []
        new_metas = []
        for i, cid in enumerate(self.corpus_ids):
            if cid not in remove_set:
                new_ids.append(self.corpus_ids[i])
                new_texts.append(self.corpus_texts[i])
                new_metas.append(self.corpus_metadatas[i])
        if new_ids:
            self.build(new_ids, new_texts, new_metas)
        else:
            self.corpus_ids = []
            self.corpus_texts = []
            self.corpus_metadatas = []
            self.bm25 = None
            self._loaded = False

    def save(self):
        """Persist index to disk. Lưu BM25 index + corpus texts riêng."""
        if not self._loaded or self.bm25 is None:
            return
        os.makedirs(self.index_path, exist_ok=True)

        # Save BM25 index (tokenized data, vocab, scores)
        # bm25s 0.3.x: save() without corpus param (chỉ lưu index)
        self.bm25.save(self.index_path)

        # Save metadata + corpus texts vào JSON riêng
        num_docs = len(self.corpus_ids)
        if hasattr(self.bm25, 'scores') and isinstance(self.bm25.scores, dict):
            self.bm25.scores['num_docs'] = num_docs
        meta_path = os.path.join(self.index_path, "bm25_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "ids": self.corpus_ids,
                "texts": self.corpus_texts,
                "metadatas": self.corpus_metadatas,
                "num_docs": num_docs
            }, f, ensure_ascii=False)

    def load(self) -> bool:
        """Load index from disk. Returns True if successful.

        bm25s 0.3.x: load() KHÔNG dùng load_corpus=True (bug trong version này).
        BM25 index tự load từ npy files. Corpus texts restore từ bm25_meta.json.
        """
        try:
            meta_path = os.path.join(self.index_path, "bm25_meta.json")
            if not os.path.exists(meta_path):
                return False

            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.corpus_ids = meta["ids"]
            self.corpus_texts = meta.get("texts", [])  # backward compat
            self.corpus_metadatas = meta["metadatas"]

            # Validate: ids và texts phải cùng độ dài
            if len(self.corpus_ids) != len(self.corpus_texts):
                # Fallback: pad texts nếu thiếu
                while len(self.corpus_texts) < len(self.corpus_ids):
                    self.corpus_texts.append("")

            # Load BM25 index (KHÔNG load_corpus=True)
            self.bm25 = bm25s.BM25()
            self.bm25.load(self.index_path, load_corpus=False)

            # bm25s 0.3.x: cần load_scores riêng + restore num_docs
            num_docs = meta.get("num_docs", len(self.corpus_ids))
            try:
                self.bm25.load_scores(self.index_path)
                if isinstance(self.bm25.scores, dict):
                    self.bm25.scores['num_docs'] = num_docs
            except Exception:
                pass

            # bm25s 0.3.x: load() không rebuild vocab_dict, cần re-index từ corpus
            tokenized_corpus = [self._tokenize(t) for t in self.corpus_texts]
            self.bm25.index(tokenized_corpus)

            self._loaded = True
            self._validate_invariant()
            return True
        except Exception as e:
            print(f"[BM25] Load error: {e}")
            return False

    def clear(self):
        """Xóa index trong memory."""
        self.corpus_ids = []
        self.corpus_texts = []
        self.corpus_metadatas = []
        self.bm25 = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded and self.bm25 is not None

    @property
    def size(self) -> int:
        return len(self.corpus_ids)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple tokenization: lowercase + split. Hỗ trợ tiếng Việt ở mức cơ bản."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()
        return [t for t in tokens if len(t) > 1]
