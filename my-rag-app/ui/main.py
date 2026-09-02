import sys
import os

# Thêm root dir vào sys.path để import các module dễ dàng
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from loaders.factory import LoaderFactory
from core.text_splitter import TextSplitterService
from core.embedding_service import OllamaEmbeddingService
from core.vector_store import ChromaVectorStore
from core.llm_service import OllamaLLMService
from services.document_service import DocumentService
from services.rag_service import RAGService
from ui.components.sidebar import render_sidebar
from ui.components.chat_message import render_chat_messages

# Cấu hình trang Streamlit
st.set_page_config(
    page_title="RAG Tra Cứu Tài Liệu Offline",
    page_icon="🔍",
    layout="wide"
)

# Khởi tạo các Service một lần duy nhất với @st.cache_resource
@st.cache_resource
def init_services():
    loader_factory = LoaderFactory()
    splitter_service = TextSplitterService()
    embedding_service = OllamaEmbeddingService()
    vector_store = ChromaVectorStore()
    llm_service = OllamaLLMService()

    document_service = DocumentService(
        loader_factory=loader_factory,
        splitter_service=splitter_service,
        embedding_service=embedding_service,
        vector_store=vector_store
    )

    rag_service = RAGService(
        embedding_service=embedding_service,
        vector_store=vector_store,
        llm_service=llm_service
    )

    return document_service, rag_service, vector_store

document_service, rag_service, vector_store = init_services()

# Render Sidebar
render_sidebar(document_service, vector_store)

# Giao diện chính
st.title("🔍 Tra Cứu Tài Liệu Offline (Local RAG)")
st.caption("Chạy hoàn toàn trên máy tính cá nhân - Bảo mật - Không dùng Internet")

# Render lịch sử tin nhắn
render_chat_messages()

# Khung nhập câu hỏi người dùng
if prompt := st.chat_input("Đặt câu hỏi về tài liệu của bạn..."):
    # 1. Hiển thị tin nhắn người dùng
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Xử lý tra cứu RAG & Stream câu trả lời
    with st.chat_message("assistant"):
        with st.spinner("Đang tìm kiếm ngữ cảnh & suy luận..."):
            history_for_rag = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]
                if m["role"] in ["user", "assistant"]
            ]
            rag_res = rag_service.query(prompt, chat_history=history_for_rag)

        # Stream từng từ
        response_placeholder = st.empty()
        full_response = ""
        for chunk in rag_res["stream"]:
            full_response += chunk
            response_placeholder.markdown(full_response + "▌")
        
        response_placeholder.markdown(full_response)

        # Hiển thị trích dẫn nguồn
        sources = rag_res["sources"]
        if sources:
            with st.expander("📌 Trích dẫn nguồn tài liệu"):
                for idx, src in enumerate(sources, 1):
                    st.markdown(f"**[{idx}] Nguồn:** `{src['file_name']}`")
                    st.caption(f'"{src["text"]}"')

    # 3. Lưu câu trả lời vào session_state
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "sources": sources
    })
