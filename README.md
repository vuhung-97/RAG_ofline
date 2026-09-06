# RAG Tra Cứu Tài Liệu Offline — PyQt6 + Ollama + ChromaDB + BM25

> Ứng dụng desktop RAG (Retrieval-Augmented Generation) chạy **100% offline** trên Windows, cho phép nạp tài liệu cá nhân/văn phòng và tra cứu bằng ngôn ngữ tự nhiên với trích dẫn nguồn chính xác.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![PyQt6](https://img.shields.io/badge/UI-PyQt6-41CD52?style=flat-square)
![Ollama](https://img.shields.io/badge/LLM-Ollama-000000?style=flat-square)
![ChromaDB](https://img.shields.io/badge/Vector-ChromaDB-ff6b6b?style=flat-square)
![BM25](https://img.shields.io/badge/Search-BM25%20%2B%20RRF-4dabf7?style=flat-square)
![Platform](https://img.shields.io/badge/Platform-Windows-0078D6?style=flat-square&logo=windows)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

**Abstract (EN):** Offline RAG desktop app for personal/office document Q&A. Built with PyQt6, Ollama (embeddinggemma:300m + qwen3:1.7b), ChromaDB and BM25 hybrid search with RRF fusion. Runs fully local without GPU, supports multi-workspace and cited answers.

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

Dự án xây dựng hệ thống **RAG tra cứu tài liệu** chạy hoàn toàn local trên laptop **không cần GPU**, không phụ thuộc cloud.

*   **Bài toán:** Người dùng nạp nhiều tài liệu (Word, Excel, PowerPoint, PDF, Text) vào các "Nhóm tài liệu" (workspace) riêng biệt, sau đó đặt câu hỏi — hệ thống truy xuất đúng đoạn liên quan và trả lời có trích dẫn `[1]`, `[2]` rõ ràng.
*   **Triết lý thiết kế:** SRP (Single Responsibility Principle) + Modular Architecture + Dependency Injection. Mỗi class chỉ làm một việc, phụ thuộc một chiều `UI → Services → Core/Retrieval → Config`.
*   **Chế độ offline:** Toàn bộ LLM/Embedding chạy qua **Ollama** tại `http://localhost:11434`. Không gửi dữ liệu ra ngoài.

---

## 2. Tính năng chính

| Nhóm | Chi tiết |
|------|----------|
| **Đa định dạng** | `.docx`, `.pdf`, `.pptx`, `.xlsx`, `.txt`, `.md`, `.doc` — chuyển qua **MarkItDown** → Markdown thống nhất (`core/markitdown_loader.py`) |
| **Chunking thông minh** | `MarkdownHeaderTextSplitter (#, ##, ###)` → `RecursiveCharacterTextSplitter` (`CHUNK_SIZE=750`, `CHUNK_OVERLAP=200`). Bảng được bọc `TABLE_START/END` và gán `table_id` để không bị cắt vụn |
| **Hybrid Search** | **Semantic (ChromaDB)** + **Keyword (BM25s)** → **RRF Fusion (k=60)** → Intent Boosting → Dedup → Neighbor Expansion |
| **Chống ảo giác** | `GuardrailValidator` + `RelevanceChecker`: chỉ trả lời từ Context, nếu thiếu trả về _"Tài liệu không đề cập đến thông tin này."_ |
| **Workspace** | Đa không gian lưu trữ (mỗi workspace = 1 ChromaDB collection). Tạo/chuyển/xóa nhóm, xóa từng file, `VACUUM` SQLite khi xóa vật lý |
| **Chat** | Streaming token theo thời gian thực, hiển thị typing indicator (hiệu ứng ba chấm đang nhảy), nút **Dừng** hủy stream, lịch sử `messages` (RAM) + `ChatLogger` JSONL |
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
│  core/reranker.py, perf_logger.py, app_settings.py      │
│  retrieval/hybrid_searcher.py, fusion.py (RRF)          │
│  retrieval/dedup.py, neighbor_expander.py               │
│  retrieval/context_builder.py, prompt_builder.py        │
│  retrieval/guardrail.py, relevance_checker.py           │
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
| `DocumentService` | Điều phối `Loader → Processor → Embed → Store` |
| `RAGService` | Điều phối `Embed Query → Hybrid Search → Context → LLM` |
| `PromptBuilder` | Chỉ build `system_content` + `messages` |
| `ui/*` | Chỉ render, không chứa business logic |

---

## 4. Cấu trúc thư mục

```text
RAG_project/
├── README.md
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
    │   ├── reranker.py           # Reranking (optional)
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
    │   ├── hybrid_searcher.py    # Semantic + BM25 → RRF
    │   ├── fusion.py             # reciprocal_rank_fusion
    │   ├── dedup.py              # deduplicate_chunks
    │   ├── neighbor_expander.py  # Mở rộng chunk lân cận
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
        ├── settings_dialog.py    # Dialog cài đặt model
        ├── workers.py            # QThread: UploadWorker, StreamWorker
        ├── styles.py             # LIGHT_THEME
        └── typing_indicator.py   # Hiệu ứng 3 chấm
```

---

## 5. Cài đặt

### 5.1. Ollama

Ứng dụng sử dụng **Ollama** để chạy mô hình LLM và Embedding hoàn toàn offline trên máy local. Ollama là nền tảng giúp đóng gói và vận hành các mô hình ngôn ngữ lớn ngay trên máy cá nhân mà không cần gửi dữ liệu ra ngoài.

Tải và cài đặt Ollama tại trang chính thức: **https://ollama.com/download**

Sau khi cài đặt, làm theo hướng dẫn trên trang Ollama để tải các model cần thiết và khởi chạy dịch vụ nền `ollama serve`.

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
5.  Sau khi xong: sidebar tự refresh danh sách `📄 File đã nạp:` — mỗi file có nút `🗑️` xóa riêng (chỉ xóa file, giữ nhóm và chat).

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
| `selected_llm` | `qwen3:1.7b` | Chọn trong `ollama list` (lọc `llm`) |
| `selected_embed` | `embeddinggemma:300m` | Chọn trong `ollama list` (lọc `embed`) |
| `num_ctx` | `8192` | Context window gửi Ollama |
| `top_k` | `6` | Số chunk cuối cùng đưa vào prompt |
| `temperature` | `0.0` | Độ ngẫu nhiên (0 = xác định/deterministic) |
| `font_size` | `10` | Áp dụng toàn app qua `QFont("Segoe UI")` |
| `enable_thinking` | `true` | Bật/tắt `<think>` của Qwen3 |
| `enable_rerank` | `true` | Bật/tắt reranking |

Nhấn **Lưu** → ghi `app_settings.json` → áp dụng font ngay + cập nhật header `LLM: ... | Context: ... | Top-K: ...`.

### 7.5. Bộ nhớ

*   `🗑️ Xóa Chat` — chỉ xóa `messages` trong RAM, không xóa DB.
*   `⚠️ Xóa Nhóm` — xóa DB vật lý như mô tả trên.

---

## 8. Cấu hình

Toàn bộ tham số tập trung tại `my-rag-app/config.py` (dataclass `Config`). Đổi ở đây sẽ áp dụng toàn pipeline (không hardcode rải rác).

| Tham số | Giá trị | Ý nghĩa |
|---------|---------|---------|
| `EMBED_MODEL` | `embeddinggemma:300m` | Model embedding Ollama |
| `LLM_MODEL` | `qwen3:1.7b` | Model LLM Ollama |
| `OLLAMA_HOST` | `http://localhost:11434` | Endpoint Ollama |
| `OLLAMA_KEEP_ALIVE` | `5m` | Giữ model trong RAM 5 phút sau idle |
| `LLM_NUM_CTX` | `8192` | Fallback mặc định — giá trị thực từ `app_settings.json` |
| `TEMPERATURE` | `0.0` | Deterministic |
| `ENABLE_THINKING` | `True` | Cho phép Qwen3 thinking |
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
| `TOP_K` | `6` | Top-K cuối cùng |
| `ENABLE_RERANK` | `True` | Bật rerank |
| `DISTANCE_THRESHOLD` | `1.10` | Lọc kết quả semantic kém |
| `CHROMA_PERSIST_DIR` | `my-rag-app/chroma_db` | Đường dẫn Chroma |
| `BM25_ENABLED` | `True` | Bật BM25 |
| `RRF_K` | `60` | Hằng số RRF |
| `ENABLE_NEIGHBOR_EXPANSION` | `True` | Mở rộng chunk lân cận |
| `NEIGHBOR_SAME_SECTION_ONLY` | `True` | Chỉ mở rộng cùng section |
| `NEIGHBOR_MAX_EXPANSION` | `1` | Số chunk mở rộng tối đa |
| `FONT_SIZE` | `10` | Cỡ chữ mặc định |

Cài đặt người dùng ghi đè được lưu tại `my-rag-app/app_settings.json`:

```json
{
  "selected_llm": "qwen3:1.7b",
  "selected_embed": "embeddinggemma:300m",
  "num_ctx": 4096,
  "top_k": 2,
  "temperature": 0.1,
  "font_size": 12,
  "enable_thinking": false,
  "enable_rerank": true
}
```

---

## 9. Pipeline RAG chi tiết

Triển khai tại `services/rag_service.py:query` (9 bước, có log thời gian từng bước):

```text
User Query
  │
  ├─ 1. IntentAnalyzer.detect_intent(query)
  │     → is_summary? (tăng top_k x5, num_ctx → 8192)
  │
  ├─ 2. OllamaEmbeddingService.embed_query(query, prefix="task: search result | query: ")
  │     → query_vector (768d)  [~t_embed ms]
  │
├─ 3. HybridSearcher.search(query, query_vector, top_k, top_k, top_k)
│     ├─ ChromaVectorStore.search_similarity(query_vector, top_k)  [semantic]
│     ├─ BM25Index.search(query, top_k)                            [keyword]
│     ├─ reciprocal_rank_fusion(semantic, bm25, k=60)               [RRF]
│     └─ Intent boosting (+0.01 nếu match chapter/heading)          [~t_search ms]
  │
  ├─ 4. deduplicate_chunks(fused_results)
  │
  ├─ 5. expand_neighbors(deduped, chunks_map, max_expansion=1, same_section_only=True)
  │     → lấy prev/next IDs từ Chroma metadata, mở rộng cùng section  [~t_neighbor ms]
  │
  ├─ 6. build_context(expanded, max_chunks=top_k, max_tokens=num_ctx-1000)
  │     → formatted_context (Markdown với [1], [2]...), merged_chunks  [~t_context ms]
  │     → format_sources_for_ui(merged_chunks)  → hiển thị bộ mở rộng nguồn
  │
  ├─ 7. has_sufficient_relevance(merged_chunks) + Guardrail check
  │     → nếu rỗng / không đủ relevance → trả về "Không tìm thấy..." (no_context=True)
  │
  ├─ 8. PromptBuilder.build_system_content(is_summary, context, num_chunks)
  │     PromptBuilder.build_messages(system, query, chat_history[-4:], enable_thinking)
  │     → messages = [system, ...history, user]  (thêm /no_think nếu tắt thinking)
  │
  └─ 9. OllamaLLMService.stream_chat(messages, model, num_ctx, temperature, think)
        → Generator[str] streaming, đo TTFT, yield từng token
        → GuardrailValidator.validate_answer(answer, context) hậu kiểm
```

**System Prompt (`retrieval/prompt_builder.py:SYSTEM_PROMPT`):**

> Chỉ dùng thông tin trong Context, không bịa đặt, mỗi dòng có nhãn `[1]..[n]` ở cuối dòng, không tự tạo khối Citations ở cuối, nếu thiếu → _"Tài liệu không đề cập đến thông tin này."_

**Guardrail (`retrieval/guardrail.py`):**

> Sau khi LLM sinh xong, GuardrailValidator kiểm tra citation hợp lệ, độ dài answer/context, và word overlap. Nếu có vấn đề → hiển thị answer gốc kèm cảnh báo ⚠️ phía dưới (không xóa answer).

| Triệu chứng | Nguyên nhân | Cách khắc phục |
|-------------|-------------|----------------|
| `Lỗi kết nối Ollama` | `ollama serve` chưa chạy | Mở Ollama app hoặc chạy lệnh `ollama serve` trong terminal, kiểm tra `http://localhost:11434/api/tags` |
| `model not found` | Chưa pull model | Tham khảo hướng dẫn tải model trên trang Ollama: https://ollama.com/library |
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
*   [ ] Reranker cross-encoder (Qwen3-Rerank) thay heuristic.
*   [ ] Export chat + sources ra PDF/Markdown.
*   [ ] Hỗ trợ thêm `.csv`, `.html` qua `CsvLoader`/`HtmlLoader` (OCP — Open-Closed Principle: chỉ thêm 1 file + 1 dòng trong factory).
*   [ ] Đóng gói installer `.exe` (PyInstaller) kèm Ollama embedded.

---

## Giấy phép & Tác giả

*   License: **MIT** (đề xuất — thay đổi nếu cần).
*   Tác giả: Nhóm CNTT14 — Dự án RAG Tra Cứu Tài Liệu.
*   Liên hệ: _điền email / GitHub_.

---

> **Ghi chú:** Toàn bộ tham số, đường dẫn và luồng xử lý trong README được trích từ mã nguồn thực tế (`config.py`, `core/*`, `services/*`, `retrieval/*`, `ui/main.py`). Không tham chiếu tài liệu kế hoạch cũ đã lỗi thời.
