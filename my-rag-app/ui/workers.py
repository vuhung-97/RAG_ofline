"""QThread workers cho background tasks: Upload/Embed, Stream LLM, Load Models."""

import os
import tempfile
from PyQt6.QtCore import QThread, pyqtSignal


class UploadWorker(QThread):
    """Worker xử lý upload + embed file trong background."""

    progress = pyqtSignal(int, str)  # (percent, status_text)
    finished = pyqtSignal(dict)  # result dict từ DocumentService
    error = pyqtSignal(str)

    def __init__(self, document_service, file_paths, file_names, embed_model):
        super().__init__()
        self.document_service = document_service
        self.file_paths = file_paths
        self.file_names = file_names
        self.embed_model = embed_model
        self._is_cancelled = False

    def run(self):
        try:
            total = len(self.file_paths)
            for idx, (fpath, fname) in enumerate(zip(self.file_paths, self.file_names)):
                if self._is_cancelled:
                    break

                self.progress.emit(
                    int((idx / total) * 100),
                    f"Đang xử lý: {fname}..."
                )

                def update_progress(current_chunk, total_chunks):
                    p = int(((idx + current_chunk / total_chunks) / total) * 100)
                    self.progress.emit(p, f"Embedding '{fname}': {current_chunk}/{total_chunks} chunks...")

                result = self.document_service.process_and_index_file(
                    file_path=fpath,
                    file_name=fname,
                    embed_model=self.embed_model,
                    progress_callback=update_progress
                )
                self.finished.emit(result)

            self.progress.emit(100, "Hoàn thành!")
        except Exception as e:
            self.error.emit(str(e))

    def cancel(self):
        self._is_cancelled = True


class StreamWorker(QThread):
    """Worker stream LLM response token by token."""

    token_received = pyqtSignal(str)
    sources_ready = pyqtSignal(list)
    answer_replaced = pyqtSignal(str)  # emit khi guardrail replace answer
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, rag_service, user_query, chat_history,
                 llm_model, embed_model, top_k, num_ctx, temperature, enable_thinking=True, enable_rerank=True):
        super().__init__()
        self.rag_service = rag_service
        self.user_query = user_query
        self.chat_history = chat_history
        self.llm_model = llm_model
        self.embed_model = embed_model
        self.top_k = top_k
        self.num_ctx = num_ctx
        self.temperature = temperature
        self.enable_thinking = enable_thinking
        self.enable_rerank = enable_rerank
        self._is_cancelled = False

    def run(self):
        try:
            result = self.rag_service.query(
                user_query=self.user_query,
                chat_history=self.chat_history,
                llm_model=self.llm_model,
                embed_model=self.embed_model,
                top_k=self.top_k,
                num_ctx=self.num_ctx,
                temperature=self.temperature,
                enable_thinking=self.enable_thinking,
                enable_rerank=self.enable_rerank
            )

            self.sources_ready.emit(result["sources"])

            full_response = ""
            for token in result["stream"]:
                if self._is_cancelled:
                    break
                self.token_received.emit(token)
                full_response += token

            if not self._is_cancelled and full_response:
                context_text = "\n".join([s.get("text", "") for s in result.get("sources", [])])
                validated = self.rag_service._validate_answer(full_response, context_text)
                if validated != full_response:
                    # Guardrail fail → replace toàn bộ answer
                    self.answer_replaced.emit(validated)

        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()

    def cancel(self):
        self._is_cancelled = True


class ModelLoaderWorker(QThread):
    """Worker tải danh sách model từ Ollama."""

    models_ready = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, host):
        super().__init__()
        self.host = host

    def run(self):
        try:
            from config import get_installed_models
            models = get_installed_models(self.host)
            self.models_ready.emit(models)
        except Exception as e:
            self.error.emit(str(e))


class FileProcessor:
    """Helper xử lý file upload từ QFileDialog."""

    @staticmethod
    def save_uploaded_files(file_paths):
        """Lưu list file paths vào temp directory, trả về (temp_paths, file_names)."""
        temp_paths = []
        file_names = []
        for path in file_paths:
            fname = os.path.basename(path)
            ext = os.path.splitext(fname)[1]
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext, dir=tempfile.gettempdir())
            with open(path, "rb") as f:
                tmp.write(f.read())
            tmp.close()
            temp_paths.append(tmp.name)
            file_names.append(fname)
        return temp_paths, file_names

    @staticmethod
    def cleanup_temp_files(paths):
        """Xóa temp files."""
        for p in paths:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
