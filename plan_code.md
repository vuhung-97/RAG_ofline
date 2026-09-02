# KẾ HOẠCH THỰC THI MÃ NGUỒN (PLAN CODE)

## 1. MỤC TIÊU VÀ NGUYÊN TẮC
- Ngôn ngữ: Python 3.10+ (App Python Streamlit duy nhất, không dùng React/Node.js để tối ưu RAM < 500MB).
- Kiến trúc: Modular Architecture + Single Responsibility Principle (SRP).
- Giao diện: Streamlit UI.

---

## 2. CHI TIẾT CÁC MODULE VÀ FILE NGUỒN CẦN VIẾT

### PHASE 1: CẤU HÌNH & LOADERS (NẠP FILE THÔ)
1. `my-rag-app/config.py` (Đã khởi tạo):
   - Đảm bảo có đủ: `EMBED_MODEL = "embeddinggemma:300m"`, `LLM_MODEL = "qwen3:1.7b"`, `LLM_NUM_CTX = 8192`, `OLLAMA_KEEP_ALIVE = "5m"`, `CHUNK_SIZE = 500`, `CHUNK_OVERLAP = 50`, `TOP_K = 4`.

2. `my-rag-app/loaders/base.py`:
   - Định nghĩa `BaseLoader(ABC)` với method `@abstractmethod def load(self, file_path: str) -> str`.

3. `my-rag-app/loaders/docx_loader.py`:
   - Class `DocxLoader(BaseLoader)`: Trích xuất text từ đoạn văn & bảng trong file `.docx`.

4. `my-rag-app/loaders/xlsx_loader.py`:
   - Class `XlsxLoader(BaseLoader)`: Đọc file Excel `.xlsx`, chuyển dữ liệu ô/sheet thành bảng Markdown (`| col1 | col2 |`).

5. `my-rag-app/loaders/pptx_loader.py`:
   - Class `PptxLoader(BaseLoader)`: Duyệt từng slide trong `.pptx` lấy toàn bộ text.

6. `my-rag-app/loaders/pdf_loader.py`:
   - Class `PdfLoader(BaseLoader)`: Đọc văn bản `.pdf` (text-native) bằng `pypdf`.

7. `my-rag-app/loaders/text_loader.py`:
   - Class `TextLoader(BaseLoader)`: Đọc file `.txt`, `.md` mã hóa UTF-8.

8. `my-rag-app/loaders/factory.py`:
   - Class `LoaderFactory`: Đăng ký map đuôi file -> Loader instance. Method `get_loader(file_path: str) -> BaseLoader`. Đảm bảo OCP (thêm loader mới không sửa code cũ).

---

### PHASE 2: CORE MODULES (THUẬT TOÁN & TƯƠNG TÁC DỊCH VỤ)
1. `my-rag-app/core/text_splitter.py`:
   - Class `TextSplitterService`: Bọc `RecursiveCharacterTextSplitter`. Phương thức `split_text(text: str, metadata: dict) -> List[Document]`.

2. `my-rag-app/core/embedding_service.py`:
   - Class `OllamaEmbeddingService`: Gọi API `http://localhost:11434/api/embed` cho model `embeddinggemma:300m`. Trả về `List[float]`.

3. `my-rag-app/core/vector_store.py`:
   - Class `ChromaVectorStore`: Bọc `chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)`. Các phương thức:
     - `add_documents(documents, embeddings, ids, metadatas)`
     - `search_similarity(query_embedding, top_k) -> List[dict]`
     - `clear_store()`

4. `my-rag-app/core/llm_service.py`:
   - Class `OllamaLLMService`: Gọi API `http://localhost:11434/api/chat` cho `qwen3:1.7b` với `options={"num_ctx": 8192}`, hỗ trợ generator streaming cho UI.

---

### PHASE 3: SERVICES MODULES (TẦNG ĐIỀU PHỐI NGHIỆP VỤ)
1. `my-rag-app/services/document_service.py`:
   - Class `DocumentService`:
     - Nhận file -> Gọi `LoaderFactory` -> Đọc text -> Gọi `TextSplitterService` -> Gọi `OllamaEmbeddingService` -> Lưu vào `ChromaVectorStore`.
     - Kiểm tra trùng lặp file (tránh embed lại file cũ).

2. `my-rag-app/services/rag_service.py`:
   - Class `RAGService`:
     - Nhận `query` -> Gọi `OllamaEmbeddingService.embed(query)` -> Tra cứu `ChromaVectorStore.search_similarity()` -> Dựng Prompt với ngữ cảnh -> Gọi `OllamaLLMService.stream_chat()`.

---

### PHASE 4: UI MODULES (GIAO DIỆN STREAMLIT)
1. `my-rag-app/ui/components/sidebar.py`:
   - Render Upload File Widget.
   - Hiển thị danh sách file đã lưu trong ChromaDB.
   - Nút "Xóa lịch sử trò chuyện" (Chỉ xóa `session_state`).
   - Nút "Xóa kho tài liệu" (Xóa ChromaDB trên đĩa).

2. `my-rag-app/ui/components/chat_message.py`:
   - Render từng tin nhắn User/Assistant.
   - Render Expander trích dẫn nguồn tài liệu (Tên file, nội dung đoạn văn bản gốc).

3. `my-rag-app/ui/main.py`:
   - Entry point: Quản lý `st.session_state` (messages, services).
   - Đảm bảo chỉ khởi tạo service 1 lần với `@st.cache_resource`.

---

## 3. THỨ TỰ THỰC HIỆN CODE & CHECKPOINT KIỂM THỬ
1. **Bước 1:** Viết file `plan_code.md` lên thư mục gốc.
2. **Bước 2:** Viết trọn bộ `loaders/` (`base.py`, 5 file loader, `factory.py`).
3. **Bước 3:** Viết `core/` (`text_splitter.py`, `embedding_service.py`, `vector_store.py`, `llm_service.py`).
4. **Bước 4:** Viết `services/` (`document_service.py`, `rag_service.py`).
5. **Bước 5:** Viết `ui/` (`sidebar.py`, `chat_message.py`, `main.py`).
6. **Bước 6:** Chạy ứng dụng bằng `streamlit run ui/main.py` và kiểm thử tích hợp.
