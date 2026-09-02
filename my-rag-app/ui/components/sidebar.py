import os
import tempfile
import streamlit as st
from services.document_service import DocumentService
from core.vector_store import ChromaVectorStore

def render_sidebar(document_service: DocumentService, vector_store: ChromaVectorStore):
    """SRP: Render thanh công cụ Sidebar (Upload, Quản lý tài liệu & Bộ nhớ)."""
    with st.sidebar:
        st.title("📚 Quản Lý Tài Liệu")
        st.markdown("---")

        # 1. Widget Upload File
        st.subheader("1. Nạp Tài Liệu Mới")
        uploaded_files = st.file_uploader(
            "Chọn file (.docx, .xlsx, .pptx, .pdf, .txt, .md)",
            type=["docx", "xlsx", "pptx", "pdf", "txt", "md"],
            accept_multiple_files=True
        )

        if uploaded_files:
            if st.button("📥 Xử Lý & Đưa Vào Kho Vector", use_container_width=True):
                progress_bar = st.progress(0)
                for idx, uploaded_file in enumerate(uploaded_files):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name

                    try:
                        res = document_service.process_and_index_file(tmp_path, uploaded_file.name)
                        if res["status"] == "success":
                            st.success(res["message"])
                        else:
                            st.info(res["message"])
                    except Exception as e:
                        st.error(f"Lỗi khi xử lý {uploaded_file.name}: {str(e)}")
                    finally:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                st.rerun()

        st.markdown("---")

        # 2. Danh sách file đã lưu trong ChromaDB
        st.subheader("2. Kho Tài Liệu Đã Nạp")
        indexed_files = vector_store.get_indexed_files()
        if indexed_files:
            for f in indexed_files:
                st.text(f"📄 {f}")
        else:
            st.caption("Chưa có tài liệu nào trong kho.")

        st.markdown("---")

        # 3. Quản lý Bộ nhớ (RAM vs Disk)
        st.subheader("3. Quản Lý Bộ Nhớ")
        
        # Nút xóa chat history (RAM)
        if st.button("🗑️ Xóa Lịch Sử Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        # Nút xóa kho dữ liệu Vector (Disk)
        if st.button("⚠️ Xóa Toàn Bộ Kho Vector", type="primary", use_container_width=True):
            vector_store.clear_store()
            st.session_state.messages = []
            st.success("Đã xóa sạch kho dữ liệu Vector!")
            st.rerun()
