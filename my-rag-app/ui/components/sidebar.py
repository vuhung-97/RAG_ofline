import os
import tempfile
import streamlit as st
from services.document_service import DocumentService
from core.vector_store import ChromaVectorStore
from config import config, get_installed_models

def render_sidebar(document_service: DocumentService, vector_store: ChromaVectorStore):
    """SRP: Render Sidebar với Quản lý Bộ nhớ đẩy lên trước, Quản lý Workspace & Cấu hình Model động."""
    with st.sidebar:
        st.title("⚙️ Cấu Hình & Quản Lý")
        st.markdown("---")

        # -------------------------------------------------------------
        # 1. QUẢN LÝ BỘ NHỚ (ĐẨY LÊN ĐẦU THEO YÊU CẦU)
        # -------------------------------------------------------------
        st.subheader("🧹 1. Quản Lý Bộ Nhớ")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Xóa Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
        with col2:
            if st.button("⚠️ Xóa Nhóm", type="primary", use_container_width=True):
                vector_store.clear_store()
                st.session_state.messages = []
                st.success(f"Đã xóa toàn bộ dữ liệu của nhóm '{vector_store.current_workspace}'!")
                st.rerun()

        st.markdown("---")

        # -------------------------------------------------------------
        # 2. CHỌN MODEL & THÔNG SỐ CẤU HÌNH ĐỘNG
        # -------------------------------------------------------------
        st.subheader("🤖 2. Cấu Hình Model Ollama")
        installed_models = get_installed_models()

        # Chọn LLM Model động
        llm_list = installed_models["llm"]
        default_llm_idx = llm_list.index(config.LLM_MODEL) if config.LLM_MODEL in llm_list else 0
        st.session_state.selected_llm = st.selectbox(
            "Chọn LLM Model:",
            options=llm_list,
            index=default_llm_idx
        )

        # Chọn Embedding Model động
        embed_list = installed_models["embed"]
        default_embed_idx = embed_list.index(config.EMBED_MODEL) if config.EMBED_MODEL in embed_list else 0
        st.session_state.selected_embed = st.selectbox(
            "Chọn Embedding Model:",
            options=embed_list,
            index=default_embed_idx
        )

        # Sliders cấu hình thông số
        with st.expander("🛠️ Cấu hình Nâng cao (Params)"):
            st.session_state.num_ctx = st.select_slider(
                "Context Window (num_ctx):",
                options=[2048, 4096, 8192, 16384, 32768],
                value=config.LLM_NUM_CTX
            )
            st.session_state.top_k = st.slider(
                "Số đoạn tra cứu (Top-K):",
                min_value=1, max_value=10, value=config.TOP_K
            )
            st.session_state.temperature = st.slider(
                "Độ sáng tạo (Temperature):",
                min_value=0.0, max_value=1.0, value=config.TEMPERATURE, step=0.1
            )
            st.session_state.chunk_size = st.number_input(
                "Kích thước Chunk (Chunk Size):",
                min_value=100, max_value=2000, value=config.CHUNK_SIZE, step=50
            )

        st.markdown("---")

        # -------------------------------------------------------------
        # 3. QUẢN LÝ NHÓM TÀI LIỆU (WORKSPACES)
        # -------------------------------------------------------------
        st.subheader("📁 3. Nhóm Tài Liệu (Workspace)")
        
        # Danh sách Workspaces
        existing_workspaces = vector_store.list_workspaces()
        if config.DEFAULT_WORKSPACE not in existing_workspaces:
            existing_workspaces.append(config.DEFAULT_WORKSPACE)

        active_ws = st.selectbox(
            "Chọn Nhóm Làm Việc Hiện Tại:",
            options=existing_workspaces,
            index=existing_workspaces.index(vector_store.current_workspace) if vector_store.current_workspace in existing_workspaces else 0
        )
        
        # Nếu đổi workspace
        if active_ws != vector_store.current_workspace:
            vector_store.set_workspace(active_ws)
            st.session_state.messages = []
            st.rerun()

        # Tạo Workspace Mới
        with st.expander("➕ Tạo Nhóm Tài Liệu Mới"):
            new_ws_name = st.text_input("Tên Nhóm Mới:", placeholder="Ví dụ: Tài liệu Kế toán")
            if st.button("Tạo Nhóm", use_container_width=True):
                if new_ws_name.strip():
                    vector_store.set_workspace(new_ws_name.strip())
                    st.session_state.messages = []
                    st.success(f"Đã tạo và chuyển sang nhóm '{new_ws_name.strip()}'!")
                    st.rerun()
                else:
                    st.warning("Vui lòng nhập tên nhóm hợp lệ.")

        st.markdown("---")

        # -------------------------------------------------------------
        # 4. NẠP TÀI LIỆU MỚI VÀO WORKSPACE HIỆN TẠI
        # -------------------------------------------------------------
        st.subheader(f"📥 4. Nạp File Vào [{vector_store.current_workspace}]")
        uploaded_files = st.file_uploader(
            "Chọn file (.docx, .xlsx, .pptx, .pdf, .txt, .md)",
            type=["docx", "xlsx", "pptx", "pdf", "txt", "md"],
            accept_multiple_files=True
        )

        if uploaded_files:
            if st.button("📥 Xử Lý & Nạp Vào Nhóm", use_container_width=True):
                progress_bar = st.progress(0)
                for idx, uploaded_file in enumerate(uploaded_files):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name

                    try:
                        res = document_service.process_and_index_file(
                            tmp_path,
                            uploaded_file.name,
                            embed_model=st.session_state.selected_embed,
                            chunk_size=st.session_state.get("chunk_size", config.CHUNK_SIZE)
                        )
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

        # -------------------------------------------------------------
        # 5. DANH SÁCH FILE TRONG WORKSPACE HIỆN TẠI
        # -------------------------------------------------------------
        st.subheader(f"📄 5. Danh Sách File [{vector_store.current_workspace}]")
        indexed_files = vector_store.get_indexed_files()
        if indexed_files:
            for f in indexed_files:
                st.text(f"📄 {f}")
        else:
            st.caption("Nhóm này chưa có tài liệu nào.")
