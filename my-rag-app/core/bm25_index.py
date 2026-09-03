"""BM25 Index using bm25s library — SRP: chỉ index và search keyword."""

import os
import re
import json
from typing import List, Dict, Any, Tuple

import bm25s

from config import config


class BM25Index:
    """BM25 keyword search index with persistence."""

    def __init__(self, index_path: str = config.BM25_INDEX_PATH):
        self.index_path = index_path
        self.corpus_ids: List[str] = []
        self.corpus_texts: List[str] = []
        self.corpus_metadatas: List[Dict] = []
        self.bm25 = None
        self._loaded = False

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

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float, Dict]]:
        """Search BM25, return [(chunk_id, score, metadata), ...]."""
        if not self._loaded or self.bm25 is None:
            return []

        tokenized_query = [self._tokenize(query)]
        results = self.bm25.retrieve(tokenized_query, k=top_k, show_progress=False)

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
        """Persist index to disk."""
        if not self._loaded or self.bm25 is None:
            return
        os.makedirs(self.index_path, exist_ok=True)

        # Save BM25 index
        self.bm25.save(self.index_path, corpus=self.corpus_texts)

        # Save metadata separately
        meta_path = os.path.join(self.index_path, "bm25_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "ids": self.corpus_ids,
                "metadatas": self.corpus_metadatas
            }, f, ensure_ascii=False)

    def load(self) -> bool:
        """Load index from disk. Returns True if successful."""
        try:
            meta_path = os.path.join(self.index_path, "bm25_meta.json")
            if not os.path.exists(meta_path):
                return False

            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.corpus_ids = meta["ids"]
            self.corpus_metadatas = meta["metadatas"]

            # Load BM25 index
            self.bm25 = bm25s.BM25()
            self.bm25.load(self.index_path, load_corpus=True)
            # Validate BM25 was actually loaded with scores
            if not hasattr(self.bm25, 'scores'):
                self.bm25 = None
                self._loaded = False
                return False
            self._loaded = True
            return True
        except Exception:
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
