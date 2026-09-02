# KẾ HOẠCH PHÁT TRIỂN ỨNG DỤNG RAG TRA CỨU TÀI LIỆU (SRP & MODULAR ARCHITECTURE)

## 1. MỤC TIÊU DỰ ÁN
- Xây dựng ứng dụng RAG (Retrieval-Augmented Generation) tra cứu tài liệu cá nhân/văn phòng.
- Chạy hoàn toàn Local trên laptop (Intel i7-1165G7, 12GB RAM, không GPU rời).
- Thiết kế theo chuẩn **SRP (Single Responsibility Principle)** và **Modular Architecture**.
- Tối ưu bộ nhớ (RAM app < 500MB, RAM Ollama + Models ~2.3GB với lựa chọn embeddinggemma:300m+qwen3:1.7b - vượt chuẩn <2GB ~300MB nhưng chấp nhận để ưu tiên chất lượng TV).

## 2. ĐỊNH DẠNG FILE HỖ TRỢ (TEXT-NATIVE ONLY)
- Word (`.docx`)
- Excel (`.xlsx`)
- PowerPoint (`.pptx`)
- PDF văn bản (`.pdf` - không bao gồm PDF scan/OCR)
- Text (`.txt`, `.md`)

## 3. CẤU TRÚC THƯ MỤC DỰ ÁN (TUÂN THỦ SRP & MODULAR)
```text
my-rag-app/
├── requirements.txt             # Khai báo thư viện phụ thuộc
├── config.py                    # Cấu hình tham số hệ thống (Models, Chunks, Paths) - SRP: chỉ chứa config
│
├── loaders/                     # [Module] Đọc & trích xuất văn bản thô - SRP: mỗi loader 1 định dạng
│   ├── __init__.py
│   ├── base.py                  # Abstract Base Class cho các Loaders (Interface)
│   ├── factory.py               # Factory Pattern - SRP: chỉ map ext -> Loader, đảm bảo OCP
│   ├── docx_loader.py           # SRP: chỉ đọc Word (.docx) -> str
│   ├── xlsx_loader.py           # SRP: chỉ đọc Excel (.xlsx) -> str
│   ├── pptx_loader.py           # SRP: chỉ đọc PowerPoint (.pptx) -> str
│   ├── pdf_loader.py            # SRP: chỉ đọc PDF text-native (.pdf) -> str
│   └── text_loader.py           # SRP: chỉ đọc Text (.txt, .md) -> str
│
├── core/                        # [Module] Xử lý thuật toán & tương tác dịch vụ - SRP: mỗi service 1 trách nhiệm
│   ├── __init__.py
│   ├── text_splitter.py         # SRP: chỉ phân đoạn văn bản thành chunks (pure function)
│   ├── embedding_service.py     # SRP: chỉ gọi Ollama Embedding (embeddinggemma:300m - user chọn)
│   ├── vector_store.py          # SRP: chỉ quản lý ChromaDB (Index, Search, Persist)
│   └── llm_service.py           # SRP: chỉ tương tác Ollama LLM (qwen3:1.7b - user chọn)
│
├── services/                    # [Module] Tầng điều phối nghiệp vụ - SRP: chỉ orchestration
│   ├── __init__.py
│   ├── document_service.py      # SRP: điều phối Đọc file -> Split -> Vector Indexing
│   └── rag_service.py           # SRP: điều phối Query -> Vector Search -> Prompting -> LLM
│
└── ui/                          # [Module] Giao diện người dùng Streamlit - SRP: chỉ render
    ├── __init__.py
    ├── components/
    │   ├── sidebar.py           # SRP: chỉ render thanh bên (Upload, Workspace)
    │   └── chat_message.py      # SRP: chỉ render khung chat & trích dẫn nguồn
    └── main.py                  # SRP: entry point + quản lý state, không chứa business logic
```

### 3.1. NGUYÊN TẮC THIẾT KẾ BẮT BUỘC (SRP & MODULAR ARCHITECTURE)

> **Yêu cầu cốt lõi:** Mỗi class chỉ thực hiện MỘT chức năng duy nhất, phân chia module rõ ràng, dễ sửa lỗi, maintain, phát triển và mở rộng.

#### A. Single Responsibility Principle (SRP) - Mỗi Class Một Nhiệm Vụ
| Module/Class | Trách nhiệm duy nhất | Cấm làm (vi phạm SRP) |
|---|---|---|
| `BaseLoader` | Định nghĩa interface `load(path)->str` | Không xử lý split/embed/DB |
| `Docx/Xlsx/Pptx/Pdf/TextLoader` | Chỉ trích xuất text thô của 1 định dạng | Không chunk, không gọi embedding |
| `LoaderFactory` | Chỉ map `.ext` -> `Loader` instance | Không đọc nội dung file |
| `TextSplitter` | Chỉ cắt text -> `List[Chunk]` | Không gọi Ollama/Chroma |
| `EmbeddingService` | Chỉ gọi `ollama.embeddings` | Không quản lý vector store |
| `VectorStore` | Chỉ `add/query/persist` ChromaDB | Không gọi LLM/Embedding |
| `LLMService` | Chỉ gọi `ollama.chat` + streaming | Không retrieval |
| `DocumentService` | Chỉ điều phối `Loader -> Splitter -> Embed -> Store` | Không chứa logic đọc file cụ thể |
| `RagService` | Chỉ điều phối `Embed Query -> Search -> Prompt -> LLM` | Không chứa logic UI |
| `ui/*` | Chỉ render Streamlit | Không chứa business logic |

**Quy tắc kiểm tra SRP:**
- Mỗi file < 150 dòng, mỗi class < 3 public methods
- Một lý do duy nhất để thay đổi class (ví dụ: thay đổi cách đọc PDF chỉ sửa `pdf_loader.py`)
- Không hardcode: mọi tham số (chunk_size, model, persist_dir) đọc từ `config.py`

#### B. Modular Architecture - Phân Chia Rõ Ràng
- **Phụ thuộc một chiều:** `ui -> services -> core/loaders -> config`. Cấm `core` import `services` hoặc `ui`.
- **Giao tiếp qua Dependency Injection:** Truyền instance qua `__init__` (ví dụ: `RagService(embedding_service, vector_store, llm_service)`), không khởi tạo chéo bên trong.
- **Interface cô lập:** `VectorStore` không biết `EmbeddingService` tồn tại.

#### C. Dễ Sửa Lỗi, Maintain, Phát Triển & Mở Rộng (OCP)
- **Mở rộng không sửa code cũ (Open-Closed Principle):** Thêm định dạng mới (ví dụ `.csv`) chỉ cần tạo `csv_loader.py: CsvLoader(BaseLoader)` + đăng ký 1 dòng trong `factory.py`, KHÔNG sửa `DocumentService` hay bất kỳ loader nào khác.
- **Thay thế linh hoạt:** Thay `ChromaDB` bằng `FAISS` chỉ sửa `vector_store.py`; thay `Ollama` bằng `LM Studio` chỉ sửa `embedding_service.py` + `llm_service.py`.
- **Lỗi cô lập:** `DocxLoader` lỗi không ảnh hưởng `PdfLoader`; test unit riêng cho từng module.
- **Tiêu chí nghiệm thu:** Thêm 1 loader mới hoặc đổi vector DB không yêu cầu sửa quá 1 file ngoài file mới.

#### D. Quản Lý Vòng Đời Model (Tối Ưu RAM & Tốc Độ)
- **Embedding (`embeddinggemma:300m` ~600-800MB) cần cho cả 2 giai đoạn:** `DocumentService` (ingest - embed chunks) **và** `RagService` (query - embed câu hỏi để vector search). Không thể tắt hẳn sau ingest.
- **Không tắt ngay sau ingest:** Giữ `OLLAMA_KEEP_ALIVE=5m` (mặc định Ollama) cho cả `EmbeddingService` và `LLMService`. Lý do: cold start `embeddinggemma:300m` ~0.7-1.2s, `qwen3:1.7b` ~3-6s trên i7-1165G7; nếu tắt ngay mỗi query sẽ chậm thêm ~5s.
- **Hành vi:** Câu hỏi đầu load 1 lần, các câu tiếp theo trong 5 phút dùng ngay (~0.1s). Sau 5 phút idle, Ollama tự unload, RAM về ~150MB. Đỉnh RAM ~2.3GB (1.4GB + 0.7GB) - vượt chuẩn `RAM Ollama + Models < 2GB` ~300MB nhưng chấp nhận theo lựa chọn user ưu tiên chất lượng TV.
- **Quyết định chốt:** Giữ `keep_alive=5m` để tiết kiệm thời gian load, không tắt ngay như phương án cũ.

## 4. LỰA CHỌN MODEL TỐI ƯU (CHẤT LƯỢNG vs RAM) - CHỐT THEO USER
- **Embedding đã chọn:** `embeddinggemma:300m` (~300M params, ~600-800MB) - Google embedding đa ngôn ngữ, chất lượng TV tốt hơn `nomic-embed-text` ~10-15%, tối ưu cho RAG tiếng Việt. Fallback nhẹ: `nomic-embed-text` (~274MB) nếu cần RAM <2GB.
- **LLM đã chọn:** `qwen3:1.7b` (~1.4GB) - chất lượng TV tốt nhất trong tầm, tổng `embeddinggemma+qwen3` ~2.3GB (vượt chuẩn <2GB ~300MB nhưng chấp nhận để ưu tiên chất lượng).
- **Cấu hình mặc định `config.py`:** `EMBED_MODEL="embeddinggemma:300m"`, `LLM_MODEL="qwen3:1.7b"`, `LLM_NUM_CTX=8192`, `OLLAMA_KEEP_ALIVE="5m"`, `OLLAMA_HOST="http://localhost:11434"`, `EMBED_DIM=768`.

## 5. TRIỂN KHAI OFFLINE & PHÂN PHỐI CHO MÁY KHÁC
- **Yêu cầu bắt buộc:** Mỗi máy phải cài `Ollama` riêng (chạy `OllamaSetup.exe` -> có `ollama serve` nền). App `my-rag-app` chỉ là client gọi `http://localhost:11434`, không tự chứa LLM.
- **Phân phối có mạng:** Gửi `OllamaSetup.exe` + `my-rag-app.zip` + hướng dẫn `ollama pull embeddinggemma:300m && ollama pull qwen3:1.7b`.
- **Phân phối KHÔNG mạng (USB):**
  1. Trên máy có mạng: `ollama pull` đủ 2 model (`embeddinggemma:300m` + `qwen3:1.7b`), copy toàn bộ `C:\Users\<User>\.ollama\models` (chứa `blobs/` + `manifests/`) ra USB (~2.1GB).
  2. Trên máy offline: Cài `OllamaSetup.exe` (không cần mạng) -> tắt Ollama tray -> copy `models` từ USB vào `C:\Users\<User>\.ollama\models` -> khởi động lại -> `ollama list` phải hiện đủ 2 model -> `ollama run qwen3:1.7b` offline thành công.
  3. Lưu ý: Cùng OS (Windows->Windows), cùng phiên bản Ollama.

## 6. QUY TRÌNH THỰC HIỆN BẰNG CODE
- **Bước 1:** Khởi tạo thư mục dự án và file `requirements.txt`, `config.py`.
- **Bước 2:** Xây dựng module `loaders/` độc lập tuân thủ SRP.
- **Bước 3:** Xây dựng module `core/` (Splitter, Embeddings, ChromaDB, Ollama LLM).
- **Bước 4:** Xây dựng module `services/` điều phối luồng nạp file & tra cứu.
- **Bước 5:** Xây dựng giao diện `ui/` với Streamlit.
- **Bước 6:** Kiểm thử tích hợp toàn bộ luồng hoạt động.
