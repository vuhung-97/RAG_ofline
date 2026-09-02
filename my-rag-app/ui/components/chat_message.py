import streamlit as st
from typing import List, Dict, Any

def render_chat_messages():
    """SRP: Render lịch sử tin nhắn trò chuyện và trích dẫn nguồn."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            # Hiển thị trích dẫn nguồn nếu có
            if "sources" in msg and msg["sources"]:
                with st.expander("📌 Trích dẫn nguồn tài liệu"):
                    for idx, src in enumerate(msg["sources"], 1):
                        st.markdown(f"**[{idx}] Nguồn:** `{src['file_name']}`")
                        st.caption(f'"{src["text"]}"')
