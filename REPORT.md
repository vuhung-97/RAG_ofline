# REPORT — RAG Pipeline Upgrade

## 1. Pipeline Trước Khi Sửa

```
User Query
    ↓
preprocess_query_text → CHỈ lấy nội dung ngoặc kép (BUG: mất ngữ cảnh)
    ↓
get_embeddings
    ↓
hybrid_search (semantic + BM25 → RRF fusion)
    - semantic_top_k = bm25_top_k = search_top_k = 6 (mọi thứ dùng chung 1 top_k)
    - BM25 index KHÔNG persist (data loss sau restart)
    - BM25 KHÔNG isolate theo workspace (query B thấy data A)
    - Intent boost KHÔNG re-sort
    ↓
deduplicate_chunks (Jaccard ≥ 0.85)
    ↓
LLMReranker.rerank → chỉ nhận ~6 candidates → gần như vô dụng
    ↓
expand_neighbors → ±1 chunk, KHÔNG consolidate → duplicate context
    ↓
_merge_table_chunks → group theo table_id (int) → merge nhầm cross-document
    ↓
build_context(max_chunks=search_top_k) → cố định, không quality-oriented
    ↓
has_sufficient_relevance → distance có thể là None (BM25-only) → TypeError
    ↓
PromptBuilder → Qwen3 1.7B (thinking) → cần strip <think> (~20 dòng code regex)
    ↓
GuardrailValidator → strip <think> lần nữa
```

**Embedding Format:** `"task: search result | document: "` — sai format EmbeddingGemma

---

## 2. Các Lỗi Đã Xác Minh

| # | Issue | Severity | Root Cause | Fix |
|---|---|---|---|---|
| 1 | BM25 Persistence — Data Loss | CRITICAL | `corpus_texts` không restore khi load | Save/load `bm25_meta.json` + `num_docs` |
| 2 | BM25 Không Isolate Workspace | CRITICAL | 1 index global cho mọi workspace | `_sanitize_workspace()` per-workspace index |
| 3 | Single Top-K Cho Mọi Thứ | CRITICAL | `search_top_k = 6` cho cả semantic, BM25, fusion, final | `SEMANTIC_TOP_K=24`, `BM25_TOP_K=24`, `FUSION_TOP_K=20`, `FINAL_TOP_K=6` |
| 4 | Embedding Document Format | CRITICAL | Sai format `"task: search result"` | `title: <title> | text: <chunk>` |
| 5 | preprocess_query_text Cắt Query | MODERATE | Regex chỉ lấy ngoặc kép | Strip whitespace only, giữ nguyên query |
| 6 | Table Identity Cross-Document | MODERATE | Group theo `table_id` (int) | Group theo `(file_name, table_id)` tuple |
| 7 | Relevance Checker None Compare | MODERATE | `distance` có thể None (BM25-only) | Check `rerank_score` first, handle `distance=None` |
| 8 | Intent Analyzer Quá Hẹp | MODERATE | Chapter extraction CHỈ cho "tóm tắt" | Extract chapter/section/article/appendix cho mọi query |
| 9 | Intent Boost Không Re-sort | MINOR | Score boost không sort lại | Sort after boost |
| 10 | Neighbor Duplicate Context | MODERATE | ±1 expansion không consolidate | Interval consolidation (merge overlapping ranges) |
| 11 | Table Expansion Không Fetch | MODERATE | Chỉ merge table chunks có sẵn | `_expand_tables()` fetch từ ChromaDB |
| 12 | Relevance Checker Không Dùng Rerank Score | MODERATE | Chỉ check rrf_score + distance | Check `rerank_score` first |
| 13 | Context Budget Machine-Matic | MODERATE | `max(1000, num_ctx-1000)` cố định | `CONTEXT_BUDGET_EVIDENCE=3500` configured |
| 14 | No Multi-turn Query Rewrite | MODERATE | Không có module | `QueryRewriter` + `FollowUpDetector` (regex/heuristic) |
| 15 | ChromaDB Distance Metric | MINOR | Default L2 | Cosine (`hf:space: cosine`) |
| 16 | No Index Versioning | MINOR | Không detect incompatible index | `index_version.json` check on startup |

---

## 3. Pipeline Sau Khi Sửa

```
User Query
    ↓
QueryRewriter.rewrite → standalone query (nếu follow-up)
    ↓
preprocess_query_text → strip whitespace, giữ nguyên nội dung
    ↓
IntentAnalyzer.detect_intent → chapter/section/article/appendix (cho mọi query)
    ↓
get_embeddings → title: <title> | text: <chunk> format
    ↓
hybrid_search
    ├─ semantic_search(top_k=24) → cosine distance
    ├─ bm25_index.search(top_k=24) → per-workspace isolated
    └─ reciprocal_rank_fusion(k=60, top_k=20)
        + intent_boost → RE-SORT after boost
    ↓
deduplicate_chunks (Jaccard ≥ 0.85)
    ↓
LLMReranker.rerank(top_k=6) → 20 candidates → reranker có ý nghĩa
    ↓
expand_neighbors → ±1, interval consolidation → KHÔNG duplicate
    ↓
_expand_tables → fetch full table từ ChromaDB
    ↓
build_context(max_chunks=6, max_tokens=3500)
    ↓
has_sufficient_relevance → check rerank_score + distance=None
    ↓
PromptBuilder → Qwen3 4B-Instruct-2507 (non-thinking) → KHÔNG cần strip <think>
    ↓
Stream response
```

---

## 4. Config Mới

| Param | Giá trị cũ | Giá trị mới | Lý do |
|---|---|---|---|
| `SEMANTIC_TOP_K` | 6 (chung) | 24 | Pool lớn cho fusion |
| `BM25_TOP_K` | 6 (chung) | 24 | Pool lớn cho fusion |
| `FUSION_TOP_K` | 6 (chung) | 20 | Reranker cần 20+ candidates |
| `FINAL_TOP_K` | 6 | 6 | Giữ nguyên |
| `CONTEXT_BUDGET_EVIDENCE` | `max(1000, num_ctx-1000)` | 3500 | Quality > quantity |
| `DISTANCE_THRESHOLD` | 1.5 (L2) | 0.8 (cosine) | Cosine metric |
| `LLM_MODEL` | `qwen3:1.7b` | `qwen3:4b-instruct-2507` | Instruct model, không thinking |
| `ENABLE_THINKING` | True | False | Instruct model |
| `TEMPERATURE` | 0.7 | 0.1 | Deterministic for QA |
| `LLM_NUM_CTX` | 2048 | 8192 | Sweet spot 12GB RAM |
| `ENABLE_PIPELINE_DEBUG` | False | False | Debug logging khi cần |

---

## 5. Re-Index Requirements

| Bước | Thay đổi | Re-embed? | Rebuild BM25? | Recreate Chroma? |
|---|---|---|---|---|
| 2-3 | BM25 persistence + isolation | No | **YES** | No |
| 5 | Embedding format | **YES** | No | No |
| 7 | Table identity (code only) | No | No | No |
| 15 | Cosine metric | No | No | **YES** |
| 17 | Qwen3 4B-Instruct (code only) | No | No | No |

**LƯU Ý:**
- Bước 5 (embedding format) và bước 15 (cosine metric) yêu cầu xóa index cũ
- KHÔNG được trộn embedding cũ và mới
- Version check sẽ detect incompatible index và in warning

---

## 6. Benchmark Results

### Unit Tests (28 tests)

| Test File | Tests | Status |
|---|---|---|
| `test_bm25_persistence.py` | 5 | PASS |
| `test_workspace_isolation.py` | 2 | PASS |
| `test_table_identity.py` | 4 | PASS |
| `test_query_preprocessing.py` | 11 | PASS |
| `test_relevance_checker.py` | 9 | PASS |
| `test_query_rewriter.py` | 8 | PASS |
| **Total** | **39** | **ALL PASS** |

### Benchmark Queries (8 categories)

Chưa chạy được vì cần Ollama running. File: `tests/benchmark_retrieval.py`

---

## 7. Những Phần Chưa Hoàn Thành

| Bước | Mô tả | Trạng thái |
|---|---|---|
| 15 | ChromaDB cosine metric | ✅ Code done, cần re-index |
| 20 | Index versioning | ✅ Code done |
| 21 | Debug logging | ✅ Partial (rag_service + hybrid_searcher) |
| 26 | Benchmark retrieval | ✅ File created, cần Ollama để chạy |
| 27 | Benchmark LLM | ❌ Chưa tạo |
| 28 | Report | ✅ Này |

### Cần Thực Hiện Bằng Tay:
1. **Xóa index cũ:** Xóa folder `chroma_db/` và `bm25_index/` trong workspace
2. **Re-index tất cả documents:** Chạy lại app → upload documents
3. **Pull model mới:** `ollama pull qwen3:4b-instruct-2507`
4. **Chạy benchmark:** `python tests/benchmark_retrieval.py`

---

## 8. Files Đã Sửa/Tạo

### Sửa:
- `config.py` — 11 params mới
- `app_settings.json` — model + rerank settings
- `core/bm25_index.py` — Rewrite: persistence + workspace isolation
- `core/app_settings.py` — DEFAULTS expanded
- `core/vector_store.py` — Cosine metric + index versioning
- `core/markitdown_loader.py` — Title extraction moves to embedding_service
- `services/rag_service.py` — Rewrite: full pipeline + debug logging
- `services/embedding_service.py` — Rewrite: title-based format
- `services/document_service.py` — Pass metadatas to embed
- `services/llm_service.py` — Rewrite: no thinking blocks
- `retrieval/hybrid_searcher.py` — Rewrite: separate top-k + intent re-sort
- `retrieval/context_builder.py` — Rewrite: (file_name, table_id) identity
- `retrieval/relevance_checker.py` — Rewrite: distance=None + rerank_score
- `retrieval/intent_analyzer.py` — Rewrite: chapter/section/article for all queries
- `retrieval/prompt_builder.py` — Rewrite: shorter prompt for instruct model
- `retrieval/guardrail.py` — Rewrite: no <think> stripping
- `retrieval/neighbor_expander.py` — Rewrite: interval consolidation
- `ui/main.py` — Workspace-aware BM25 + no <think> regex
- `ui/workers.py` — Removed <think> regex
- `ui/settings_dialog.py` — Updated label

### Tạo mới:
- `retrieval/query_rewriter.py` — FollowUpDetector + QueryRewriter
- `tests/test_bm25_persistence.py` — 5 tests
- `tests/test_workspace_isolation.py` — 2 tests
- `tests/test_table_identity.py` — 4 tests
- `tests/test_query_preprocessing.py` — 11 tests
- `tests/test_relevance_checker.py` — 9 tests
- `tests/test_query_rewriter.py` — 8 tests
- `tests/benchmark_retrieval.py` — Benchmark script
- `temp/kehoach.md` — Implementation plan
