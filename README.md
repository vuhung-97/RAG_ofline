# RAG Tra Cứu Tài Liệu Offline — PyQt6 + Ollama + ChromaDB + BM25

> Ứng dụng desktop RAG (Retrieval-Augmented Generation) chạy **100% offline** trên Windows, cho phép nạp tài liệu cá nhân/văn phòng và tra cứu bằng ngôn ngữ tự nhiên với trích dẫn nguồn chính xác.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![PyQt6](https://img.shields.io/badge/UI-PyQt6-41CD52?style=flat-square)
![Ollama](https://img.shields.io/badge/LLM-Ollama-000000?style=flat-square)
![ChromaDB](https://img.shields.io/badge/Vector-ChromaDB-ff6b6b?style=flat-square)
![BM25](https://img.shields.io/badge/Search-BM25%20%2B%20RRF-4dabf7?style=flat-square)
![Platform](https://img.shields.io/badge/Platform-Windows-0078D6?style=flat-square&logo=windows)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

**Abstract (EN):** Offline RAG desktop app for personal/office document Q&A. Built with PyQt6, Ollama (`embeddinggemma:300m` + `qwen3:4b-instruct`), ChromaDB and BM25 hybrid search with RRF fusion. Runs fully local without GPU, supports multi-workspace and cited answers.

---

## Mục lục

- [1. Giới thiệu](#1-giới-thiệu)
- [2. Tính năng chính](#2-tính-năng-chính)
- [3. Kiến trúc](#3-kiến-trúc)
- [4. Cấu trúc thư mục](#4-cấu-trúc-thư-mục)
- [5. Cài đặt](#5-cài-đặt)
- [6. Chạy ứng dụng](#6-chạy-ứng-dụng)
- [7. Hướng dẫn sử dụng](#7-hướng-dẫn-sử-dụng)
- [8. Cấu hình](#8-cấu-hình)
- [9. Pipeline RAG chi tiết](#9-pipeline-rag-chi-tiết)
- [10. Xử lý sự cố](#10-xử-lý-sự-cố)
- [11. Hạn chế & Roadmap](#11-hạn-chế--roadmap)

---

## 1. Giới thiệu

Dự án xây dựng hệ thống **RAG tra cứu tài liệu** chạy hoàn toàn local trên máy cá nhân **không bắt buộc GPU**, không phụ thuộc cloud.

*   **Bài toán:** Người dùng nạp nhiều tài liệu (Word, Excel, PowerPoint, PDF, Text) vào các "Nhóm tài liệu" (workspace) riêng biệt, sau đó đặt câu hỏi — hệ thống truy xuất đúng đoạn liên quan và trả lời có trích dẫn `[1]`, `[2]` rõ ràng.
*   **Triết lý thiết kế:** SRP (Single Responsibility Principle) + Modular Architecture + Dependency Injection. Mỗi class chỉ làm một việc, phụ thuộc một chiều `UI → Services → Core/Retrieval → Config`.
*   **Chế độ offline:** Toàn bộ LLM/Embedding chạy qua **Ollama** tại `http://localhost:11434`. Không gửi dữ liệu ra ngoài.

---

## 2. Tính năng chính

| Nhóm | Chi tiết |
|------|----------|
| **Đa định dạng** | `.docx`, `.pdf`, `.pptx`, `.xlsx`, `.txt`, `.md`, `.doc` — chuyển qua **MarkItDown** → Markdown thống nhất (`core/markitdown_loader.py`) |
| **Chunking thông minh** | `MarkdownHeaderTextSplitter (#, ##, ###)` → `RecursiveCharacterTextSplitter` (`CHUNK_SIZE=750`, `CHUNK_OVERLAP=200`). Bảng được bọc `TABLE_START/END` và gán `table_id` để không bị cắt vụn |
| **Hybrid Search & Fusion** | **Semantic (ChromaDB)** + **Keyword (BM25s)** → **RRF Fusion (k=60)** → Intent Boosting → Dedup → LLM Reranking → Neighbor & Table Expansion |
| **Query Rewrite & Follow-up** | Tự động nhận diện câu hỏi nối tiếp (Follow-up) và viết lại query tối ưu trước khi truy xuất (`retrieval/query_rewriter.py`) |
| **Chống ảo giác** | `GuardrailValidator` + `RelevanceChecker`: chỉ trả lời từ Context, nếu thiếu trả về _"Không tìm thấy thông tin phù hợp trong tài liệu được cung cấp."_ |
| **Workspace** | Đa không gian lưu trữ (mỗi workspace = 1 ChromaDB collection). Tạo/chuyển/xóa nhóm, xóa từng file, `VACUUM` SQLite khi xóa vật lý |
| **Chat & UI** | Streaming token theo thời gian thực, hiển thị typing indicator (hiệu ứng ba chấm đang nhảy), nút **Dừng** hủy stream, lịch sử `messages` (RAM) + `ChatLogger` JSONL |
| **Cài đặt động** | Đổi LLM/Embedding model, `num_ctx`, `top_k`, `temperature`, `font_size`, `thinking`, `rerank` — persist vào `app_settings.json` |
| **Hiệu năng** | Batch embedding 16, `keep_alive=5m` (tránh cold start 3-6s), đo TTFT (Time To First Token) và thời gian từng bước pipeline |

---

## 3. Kiến trúc

```
┌─────────────────────────────────────────────────────────┐
│ UI (PyQt6)                                              │
│  ui/main.py  → MainWindow, QSplitter, header, statusBar │
│  ui/sidebar.py, chat_area.py, chat_input.py, workers.py │
└──────────────────────┬──────────────────────────────────┘
                       │ Dependency Injection
┌──────────────────────▼──────────────────────────────────┐
│ Services (Orchestration)                                │
│  services/document_service.py  Ingest pipeline          │
│  services/rag_service.py        Query pipeline          │
│  services/embedding_service.py  Ollama /api/embed       │
│  services/llm_service.py        Ollama /api/chat stream │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│ Core + Retrieval                                        │
│  core/markitdown_loader.py  File → Markdown → Chunks    │
│  core/chunk_processor.py    chunk_id (MD5) + prev/next  │
│  core/vector_store.py       ChromaVectorStore           │
│  core/bm25_index.py         BM25Index (bm25s)           │
│  core/reranker.py           LLMReranker (Reranking)     │
│  core/perf_logger.py, app_settings.py, chat_logger.py   │
│  retrieval/query_rewriter.py Follow-up & Rewrite Query  │
│  retrieval/hybrid_searcher.py, fusion.py (RRF)          │
│  retrieval/dedup.py, neighbor_expander.py               │
│  retrieval/context_builder.py, prompt_builder.py        │
│  retrieval/guardrail.py, relevance_checker.py           │
│  retrieval/intent_analyzer.py                           │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│ Config                                                  │
│  config.py  → Config dataclass (mọi tham số tập trung)  │
└─────────────────────────────────────────────────────────┘
```

**Nguyên tắc SRP đã áp dụng:**

| Class | Một trách nhiệm duy nhất |
|-------|---------------------------|
| `MarkItDownLoader` | Chỉ `load_and_chunk(path, metadata) → List[chunk]` |
| `ChunkProcessor` | Chỉ gán `chunk_id` + `previous/next_chunk_id` |
| `OllamaEmbeddingService` | Chỉ gọi `/api/embed` với prefix asymmetric |
| `ChromaVectorStore` | Chỉ `add/query/delete` ChromaDB |
| `BM25Index` | Chỉ `build/search/save/load` BM25 |
| `QueryRewriter` | Chỉ kiểm tra follow-up và làm rõ query dựa theo lịch sử |
| `LLMReranker` | Chỉ đánh giá và sắp xếp lại thứ tự liên quan của các chunks |
| `DocumentService` | Điều phối `Loader → Processor → Embed → Store` |
| `RAGService` | Điều phối `Query Rewrite → Hybrid Search → Rerank → Context → LLM` |
| `PromptBuilder` | Chỉ build `system_content` + `messages` |
| `ui/*` | Chỉ render, không chứa business logic |

---

## 4. Cấu trúc thư mục

```text
RAG_project/
├── README.md
├── REPORT.md                     # Báo cáo kỹ thuật chi tiết
├── .gitignore
└── my-rag-app/
    ├── config.py                 # Tham số hệ thống duy nhất
    ├── requirements.txt          # Dependencies
    ├── app_settings.json         # Persist cài đặt người dùng
    ├── run_app.bat               # Script chạy 1 click trên Windows
    ├── chroma_db/                # Persistent ChromaDB (ignored)
    ├── bm25_index/               # Persistent BM25 index (ignored)
    ├── logs/                     # session_*.jsonl
    │
    ├── core/
    │   ├── markitdown_loader.py  # File → Markdown → chunks, xử lý bảng
    │   ├── chunk_processor.py    # MD5 chunk_id + liên kết prev/next
    │   ├── vector_store.py       # ChromaVectorStore multi-workspace
    │   ├── bm25_index.py         # BM25Index (bm25s)
    │   ├── reranker.py           # LLMReranker
    │   ├── app_settings.py       # Load/save JSON
    │   ├── chat_logger.py        # Ghi log hội thoại
    │   └── perf_logger.py        # Đo thời gian pipeline
    │
    ├── services/
    │   ├── document_service.py   # Ingest orchestration
    │   ├── rag_service.py        # Query orchestration
    │   ├── embedding_service.py  # Ollama embedding + prefix + batch
    │   └── llm_service.py        # Ollama chat streaming
    │
    ├── retrieval/
    │   ├── query_rewriter.py     # Nhận diện follow-up & rewrite query
    │   ├── hybrid_searcher.py    # Semantic + BM25 → RRF
    │   ├── fusion.py             # reciprocal_rank_fusion
    │   ├── dedup.py              # deduplicate_chunks
    │   ├── neighbor_expander.py  # Mở rộng chunk lân cận cùng section
    │   ├── context_builder.py    # Budget max_tokens, format sources
    │   ├── prompt_builder.py     # SYSTEM_PROMPT / SUMMARY_PROMPT
    │   ├── guardrail.py          # Validate chống bịa đặt
    │   ├── relevance_checker.py  # has_sufficient_relevance
    │   └── intent_analyzer.py    # detect_intent (summary/chapter)
    │
    └── ui/
        ├── main.py               # MainWindow entry point
        ├── sidebar.py            # Workspace + file list + upload
        ├── chat_area.py          # Render bubble + sources + streaming
        ├── chat_input.py         # Ô nhập + nút Gửi/Dừng
        ├── chat_message.py       # Single message widget
        ├── settings_dialog.py    # Dialog cài đặt model & tham số RAG
        ├── workers.py            # QThread: UploadWorker, StreamWorker
        ├── styles.py             # LIGHT_THEME
        └── typing_indicator.py   # Hiệu ứng 3 chấm
```

---

## 5. Cài đặt

### 5.1. Ollama

Ứng dụng sử dụng **Ollama** để chạy mô hình LLM và Embedding hoàn toàn offline trên máy local.

Tải và cài đặt Ollama tại trang chính thức: **https://ollama.com/download**

Sau khi cài đặt, tải các mô hình khuyến nghị bằng terminal:
```bash
ollama pull qwen3:4b-instruct
ollama pull embeddinggemma:300m
```
Vận hành dịch vụ nền `ollama serve`.

### 5.2. Cài ứng dụng

```powershell
# Clone repo
git clone <repo-url>
cd RAG_project\my-rag-app

# Tạo venv
python -m venv venv
.\venv\Scripts\activate

# Cài dependencies
pip install -r requirements.txt
```

> `venv/`, `chroma_db/`, `bm25_index/` đã được `.gitignore`, không commit.

---

## 6. Chạy ứng dụng

### Cách 1 — Double click 

```powershell
# Trong my-rag-app/
.\run_app.bat
```

`run_app.bat` tự `activate venv` và chạy `python ui\main.py`.

### Cách 2 — Dòng lệnh

```powershell
.\venv\Scripts\activate
python ui\main.py
# hoặc từ root
python my-rag-app\ui\main.py
```

---

## 7. Hướng dẫn sử dụng

### 7.1. Nhóm tài liệu (Workspace)

*   **Chọn nhóm:** Dropdown ở sidebar (mặc định `Tài liệu chung`).
*   **Thêm nhóm:** `➕ Thêm Nhóm Mới` → nhập tên → tự chuyển sang nhóm mới.
*   **Xóa nhóm:** `⚠️ Xóa Nhóm` → xác nhận Yes/No → xóa vật lý collection + folder UUID trong `chroma_db/` + `VACUUM` SQLite. Tên nhóm biến mất khỏi dropdown.
*   Mỗi nhóm là một **Chroma collection** riêng biệt, BM25 index được rebuild theo nhóm hiện tại.

### 7.2. Nạp tài liệu

1.  Nhấn `📥 Chọn File Để Nạp` → chọn nhiều file `.docx/.pdf/.pptx/.xlsx/.txt/.md`.
2.  Progress bar hiển thị `%` và `current/total` batch embedding.
3.  Tránh nạp trùng: nếu `file_name` đã tồn tại trong nhóm → báo `skipped`.
4.  File rỗng / không trích được text → báo `warning`.
5.  Sau khi xong: sidebar tự refresh danh sách `📄 File đã nạp:` — mỗi file có nút `🗑️` xóa riêng.

**Luồng ingest (`services/document_service.py:process_and_index_file`):**

```
File → MarkItDown → Markdown → preprocess_tables → MarkdownHeaderSplitter
     → RecursiveCharacterSplitter → ChunkProcessor (MD5 + prev/next)
     → embed_documents (prefix document + batch 16) → ChromaDB + BM25 → save
```

### 7.3. Đặt câu hỏi

*   Nhập câu hỏi ở ô dưới cùng → `Gửi` (Enter).
*   Hiển thị **typing indicator** (hiệu ứng ba chấm đang nhảy) cho tới khi token đầu tiên về (TTFT).
*   Token được stream theo thời gian thực vào bubble của assistant. Nút chuyển thành **Dừng** — nhấn để hủy `StreamWorker` và xóa câu hỏi khỏi history.
*   Khi trích dẫn, mỗi ý có nhãn `[1]`, `[2]` trỏ tới bộ mở rộng nguồn (tên file + chunk text gốc) trong `chat_message.py`.
*   Nếu không đủ ngữ cảnh liên quan → trả lời cố định: _"Không tìm thấy thông tin phù hợp trong tài liệu được cung cấp."_

### 7.4. Cài đặt Model

Nhấn `⚙️ Cài Đặt Model` → dialog cho phép đổi:

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `selected_llm` | `qwen3:4b-instruct` | Chọn trong `ollama list` (lọc `llm`) |
| `selected_embed` | `embeddinggemma:300m` | Chọn trong `ollama list` (lọc `embed`) |
| `num_ctx` | `8192` | Context window gửi Ollama |
| `top_k` | `6` | Số chunk cuối cùng đưa vào prompt |
| `temperature` | `0.1` | Độ ngẫu nhiên (0.1 = deterministic) |
| `font_size` | `12` | Áp dụng toàn app qua `QFont("Segoe UI")` |
| `enable_thinking` | `false` | Bật/tắt `<think>` của Qwen3 |
| `enable_rerank` | `true` | Bật/tắt LLM reranking |

Nhấn **Lưu** → ghi `app_settings.json` → áp dụng font ngay + cập nhật header `LLM: ... | Context: ... | Top-K: ...`.

---

## 8. Cấu hình

Toàn bộ tham số tập trung tại `my-rag-app/config.py` (dataclass `Config`). Đổi ở đây sẽ áp dụng toàn pipeline (không hardcode rải rác).

| Tham số | Giá trị | Ý nghĩa |
|---------|---------|---------|
| `EMBED_MODEL` | `embeddinggemma:300m` | Model embedding Ollama |
| `LLM_MODEL` | `qwen3:4b-instruct` | Model LLM Ollama |
| `OLLAMA_HOST` | `http://localhost:11434` | Endpoint Ollama |
| `OLLAMA_KEEP_ALIVE` | `5m` | Giữ model trong RAM 5 phút sau idle |
| `LLM_NUM_CTX` | `8192` | Fallback mặc định — giá trị thực từ `app_settings.json` |
| `TEMPERATURE` | `0.1` | Low temperature cho độ chính xác cao |
| `ENABLE_THINKING` | `False` | Tắt thinking cho model instruct |
| `EMBED_QUERY_PREFIX` | `task: search result | query: ` | Prefix asymmetric cho query |
| `EMBED_DOC_PREFIX` | `task: search result | document: ` | Prefix asymmetric cho document |
| `EMBED_DIMENSION` | `768` | Chiều embedding |
| `EMBED_BATCH_SIZE` | `16` | Batch embed documents |
| `CHAT_HISTORY_LIMIT` | `4` | Số tin nhắn lịch sử đưa vào prompt |
| `INTENT_BOOST_SCORE` | `0.01` | Điểm cộng khi match chapter/heading |
| `GUARDRAIL_MIN_WORD_OVERLAP` | `0.3` | Ngưỡng overlap từ guardrail |
| `CHUNK_SIZE` | `750` | Kích thước chunk (ký tự) |
| `CHUNK_OVERLAP` | `200` | Overlap giữa chunks |
| `TABLE_MAX_CHARS` | `8000` | Max ký tự cho 1 bảng sau merge |
| `SEMANTIC_TOP_K` | `24` | Số chunk Semantic retrieval lấy |
| `BM25_TOP_K` | `24` | Số chunk BM25 retrieval lấy |
| `FUSION_TOP_K` | `20` | Số chunk sau RRF fusion |
| `FINAL_TOP_K` | `6` | Top-K cuối cùng đưa vào context |
| `ENABLE_RERANK` | `True` | Bật LLM rerank |
| `DISTANCE_THRESHOLD` | `1.4` | Lọc kết quả semantic Cosine distance |
| `CHROMA_PERSIST_DIR` | `my-rag-app/chroma_db` | Đường dẫn Chroma |
| `BM25_ENABLED` | `True` | Bật BM25 |
| `RRF_K` | `60` | Hằng số RRF |
| `ENABLE_NEIGHBOR_EXPANSION` | `True` | Mở rộng chunk lân cận |
| `NEIGHBOR_SAME_SECTION_ONLY` | `True` | Chỉ mở rộng cùng section |
| `NEIGHBOR_MAX_EXPANSION` | `1` | Số chunk mở rộng tối đa |
| `CONTEXT_BUDGET_EVIDENCE` | `3500` | Token budget cho evidence context |

Cài đặt người dùng ghi đè được lưu tại `my-rag-app/app_settings.json`:

```json
{
  "selected_llm": "qwen3:4b-instruct",
  "selected_embed": "embeddinggemma:300m",
  "num_ctx": 8192,
  "top_k": 6,
  "temperature": 0.1,
  "font_size": 12,
  "enable_thinking": false,
  "enable_rerank": true,
  "semantic_top_k": 24,
  "bm25_top_k": 24,
  "fusion_top_k": 20,
  "final_top_k": 6
}
```

---

## 9. Pipeline RAG chi tiết

Triển khai tại `services/rag_service.py:query` (11 bước, có log thời gian từng bước):

```text
User Query
  │
  ├─ 0. FollowUpDetector & QueryRewriter
  │     → Nếu là câu hỏi nối tiếp, viết lại query đầy đủ ngữ cảnh từ lịch sử chat.
  │
  ├─ 1. IntentAnalyzer.detect_intent(query)
  │     → is_summary? chapter/section match?
  │
  ├─ 2. OllamaEmbeddingService.embed_query(query, prefix="task: search result | query: ")
  │     → query_vector (768d)  [~t_embed ms]
  │
  ├─ 3. HybridSearcher.search(query, query_vector)
  │     ├─ ChromaVectorStore.search_similarity (Semantic top-24)
  │     ├─ BM25Index.search (Keyword top-24)
  │     ├─ reciprocal_rank_fusion (RRF k=60 -> top-20)
  │     └─ Intent boosting (+0.01 nếu match chapter/heading)  [~t_search ms]
  │
  ├─ 4. deduplicate_chunks(fused_results)
  │
  ├─ 5. LLMReranker.rerank(query, deduped, top_k=6)  [Skip nếu là summary]
  │
  ├─ 6. expand_neighbors(deduped, chunks_map, max_expansion=1, same_section_only=True)
  │     → Lấy prev/next IDs từ Chroma metadata, mở rộng cùng section  [~t_neighbor ms]
  │
  ├─ 7. _expand_tables(expanded)
  │     → Gom trọn vẹn các chunks thuộc cùng bảng dài
  │
  ├─ 8. build_context(final_chunks, max_tokens=3500)
  │     → formatted_context (Markdown với [1], [2]...), merged_chunks  [~t_context ms]
  │
  ├─ 9. has_sufficient_relevance(merged_chunks)
  │     → Nếu rỗng / không đủ độ liên quan → Trả về thông báo không tìm thấy
  │
  ├─ 10. PromptBuilder.build_system_content & build_messages
  │      → Tạp prompt chuẩn mực có trích dẫn [1]..[n]
  │
  └─ 11. OllamaLLMService.stream_chat(...)
         → Generator[str] streaming token theo thời gian thực
         → GuardrailValidator.validate_answer(...) hậu kiểm chống bịa đặt
```

---

## 10. Xử lý sự cố

| Triệu chứng | Nguyên nhân | Cách khắc phục |
|-------------|-------------|----------------|
| `Lỗi kết nối Ollama` | `ollama serve` chưa chạy | Mở Terminal và chạy lệnh `ollama serve`, kiểm tra `http://localhost:11434/api/tags` |
| `model not found` | Chưa pull model | Chạy `ollama pull qwen3:4b-instruct` và `ollama pull embeddinggemma:300m` |
| Timeout embedding (180s) | File quá lớn / batch quá nặng | Giảm `EMBED_BATCH_SIZE` trong `config.py`, chia nhỏ file |
| Timeout LLM (120s) | `num_ctx` quá lớn / prompt quá dài | Giảm `num_ctx` hoặc `top_k` trong Settings |
| `Chroma lock` / không xóa được collection | DB đang mở ở process khác | Đóng app, xóa `chroma.sqlite3-journal` nếu kẹt, khởi động lại |
| BM25 không trả kết quả | Index chưa build / chưa load | Xóa `bm25_index/` → app tự rebuild từ Chroma khi nạp file mới, hoặc gọi `DocumentService.rebuild_bm25_index()` |
| RAM cao (~2.3GB) | Cả 2 model cùng loaded | Đợi 5 phút idle → Ollama tự unload về ~150MB (`keep_alive=5m`) |
| Trả lời _"Không tìm thấy..."_ liên tục | `DISTANCE_THRESHOLD` quá chặt / chunking sai | Giảm `DISTANCE_THRESHOLD`, kiểm tra file có text-native không (PDF scan không hỗ trợ) |
| Font quá nhỏ/lớn | `font_size` trong JSON | Đổi trong Settings → Lưu, hoặc sửa `app_settings.json` |

---

## 11. Hạn chế & Roadmap

**Hạn chế hiện tại:**

*   Chỉ hỗ trợ **PDF text-native**, không OCR PDF scan/ảnh.
*   Bảng Excel/PowerPoint phức tạp có thể mất định dạng sau MarkItDown (đã xử lý bằng `TABLE_START/END` và `TABLE_MAX_CHARS=8000`).
*   Tokenize BM25 đơn giản (lowercase + split), chưa tối ưu sâu cho tiếng Việt có dấu.
*   Chưa có đánh giá tự động độ chính xác citation.

**Roadmap gợi ý:**

*   [ ] Thêm OCR (Tesseract / PaddleOCR) cho PDF scan.
*   [ ] Reranker cross-encoder thay heuristic.
*   [ ] Export chat + sources ra PDF/Markdown.
*   [ ] Hỗ trợ thêm `.csv`, `.html` qua `CsvLoader`/`HtmlLoader` (OCP — Open-Closed Principle).
*   [ ] Đóng gói installer `.exe` (PyInstaller) kèm Ollama embedded.

---

## Giấy phép & Tác giả

*   License: **MIT**.
*   Tác giả: Nhóm CNTT14 — Dự án RAG Tra Cứu Tài Liệu.

---

> **Ghi chú:** Toàn bộ tham số, đường dẫn và luồng xử lý trong README được trích từ mã nguồn thực tế (`config.py`, `core/*`, `services/*`, `retrieval/*`, `ui/main.py`).
