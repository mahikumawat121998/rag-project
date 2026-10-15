import streamlit as st
import requests
import os
import json
from typing import List, Dict, Any

# Page configuration
st.set_page_config(
    page_title="NimbusDesk AI - Enterprise Policy RAG",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Base URL (FastAPI)
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

# Custom CSS for modern styling
st.markdown("""
<style>
    /* Main container styling */
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #64748b;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    /* Stat cards */
    .stat-card {
        background-color: #f8fafc;
        border-radius: 8px;
        padding: 12px;
        border-left: 4px solid #3b82f6;
        margin-bottom: 10px;
    }
    /* Quick prompt button styling */
    div.stButton > button.prompt-btn {
        width: 100%;
        text-align: left;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        background-color: #f8fafc;
        padding: 8px 12px;
        font-size: 0.9rem;
    }
    /* Source citation tag */
    .source-badge {
        display: inline-block;
        background-color: #eff6ff;
        color: #1d4ed8;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 6px;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions to talk to FastAPI Backend
def check_api_health():
    try:
        res = requests.get(f"{API_BASE_URL}/health", timeout=2)
        return res.status_code == 200
    except Exception:
        return False

def get_chroma_status():
    try:
        res = requests.get(f"{API_BASE_URL}/api/chat/status", timeout=2)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return None

def get_indexed_documents():
    try:
        res = requests.get(f"{API_BASE_URL}/api/documents", timeout=2)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return None

def upload_document_to_api(uploaded_file):
    try:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type or "application/octet-stream")}
        res = requests.post(f"{API_BASE_URL}/api/documents/upload", files=files, timeout=40)
        if res.status_code == 200:
            return True, res.json()
        return False, res.json().get("detail", "Upload failed")
    except Exception as e:
        return False, str(e)

def ask_question(question: str, chat_history: list = None):
    try:
        history_payload = []
        if chat_history:
            for m in chat_history:
                if m.get("content"):
                    history_payload.append({"role": m.get("role", "user"), "content": m.get("content")})
        payload = {"question": question, "chat_history": history_payload}
        res = requests.post(f"{API_BASE_URL}/api/chat", json=payload, timeout=60)
        if res.status_code == 200:
            return res.json()
        return {"answer": f"API Error: {res.text}", "sources": []}
    except Exception as e:
        return {"answer": f"Connection Error: Could not reach FastAPI backend at {API_BASE_URL}. Ensure uvicorn is running.\n\nDetails: {str(e)}", "sources": []}

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I am **NimbusDesk AI**, your internal policy and documentation assistant.\n\nAsk me anything about leave policies, remote work, reimbursements, IT security, or upload new guidelines in the sidebar!",
            "sources": []
        }
    ]

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/artificial-intelligence.png", width=90)
    st.title("Nimbus Hub")

    # API Status Check
    is_online = check_api_health()
    if is_online:
        st.success("🟢 Backend API: Connected", icon="✅")
    else:
        st.error("🔴 Backend API: Offline", icon="⚠️")
        st.caption("Start backend with: `uvicorn backend.app.main:app --port 8000`")

    # Live Metrics
    status_data = get_chroma_status()
    total_vectors = status_data.get("total_vectors", 0) if status_data else 0
    st.metric(label="ChromaDB Vectors", value=f"{total_vectors} chunks")

    st.divider()

    # Document Uploader Section
    st.subheader("📤 Ingest New Document")
    uploaded_file = st.file_uploader(
        "Upload PDF, DOCX, or TXT",
        type=["pdf", "docx", "txt", "md"],
        help="Upload new policy documents to index into ChromaDB in real-time."
    )

    if uploaded_file is not None:
        if st.button("🚀 Upload & Ingest into RAG", use_container_width=True, type="primary"):
            with st.spinner(f"Ingesting '{uploaded_file.name}' into ChromaDB..."):
                success, result = upload_document_to_api(uploaded_file)
                if success:
                    st.success(f"✅ Ingested {result.get('chunks_added', 0)} chunks!")
                    st.toast(f"Document '{uploaded_file.name}' indexed successfully!")
                    st.rerun()
                else:
                    st.error(f"Failed to ingest: {result}")

    st.divider()

    # View Ingested Documents
    with st.expander("📚 Indexed Policy Documents"):
        docs_data = get_indexed_documents()
        if docs_data and docs_data.get("documents"):
            for d in docs_data["documents"]:
                st.markdown(f"• **{d['filename']}** (`{d['chunk_count']}` chunks)")
        else:
            st.caption("No document breakdown available.")

    # Reset Chat
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Chat history cleared. How can I help you today?",
                "sources": []
            }
        ]
        st.rerun()

# ----------------- MAIN CHAT UI -----------------
st.markdown('<div class="main-title">NimbusDesk AI Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Semantic Enterprise RAG powered by Google Gemini and ChromaDB</div>', unsafe_allow_html=True)

# Quick Prompts
col1, col2, col3, col4 = st.columns(4)
quick_prompt = None

with col1:
    if st.button("📅 Annual Leave Rules", use_container_width=True):
        quick_prompt = "How many days of paid annual leave do full-time employees receive?"
with col2:
    if st.button("🏠 Remote Work Hours", use_container_width=True):
        quick_prompt = "What are the core working hours and attendance rules for remote employees?"
with col3:
    if st.button("☕ Meal Reimbursement", use_container_width=True):
        quick_prompt = "What is the reimbursement limit for client meals and team dinners?"
with col4:
    if st.button("🔒 Password & 2FA Policy", use_container_width=True):
        quick_prompt = "What are the IT security guidelines for passwords and two-factor authentication?"

# Display existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "🤖"):
        st.markdown(msg["content"])
        
        # Display sources if available
        if msg.get("sources"):
            with st.expander("🔍 Cited Sources & Page Numbers"):
                for idx, src in enumerate(msg["sources"], 1):
                    source_name = src.get("source", "Unknown")
                    page = src.get("page")
                    page_badge = f"<span class='source-badge'>Page {page}</span>" if page else ""
                    st.markdown(f"**Source #{idx}:** `{source_name}` {page_badge}", unsafe_allow_html=True)
                    st.caption(f"\"{src.get('content_preview', '')}\"")

# Handle input (either from chat_input or quick prompt button)
user_query = st.chat_input("Ask a question about company policies...")
prompt_to_process = quick_prompt or user_query

if prompt_to_process:
    # 1. Append user query
    st.session_state.messages.append({"role": "user", "content": prompt_to_process, "sources": []})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt_to_process)

    # 2. Query FastAPI Backend
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Searching company documents and generating answer..."):
            prior_history = st.session_state.messages[:-1]
            response = ask_question(prompt_to_process, chat_history=prior_history)
            answer_text = response.get("answer", "No response generated.")
            sources = response.get("sources", [])

            st.markdown(answer_text)

            if sources:
                with st.expander("🔍 Cited Sources & Page Numbers"):
                    for idx, src in enumerate(sources, 1):
                        source_name = src.get("source", "Unknown")
                        page = src.get("page")
                        page_badge = f"<span class='source-badge'>Page {page}</span>" if page else ""
                        st.markdown(f"**Source #{idx}:** `{source_name}` {page_badge}", unsafe_allow_html=True)
                        st.caption(f"\"{src.get('content_preview', '')}\"")

    # 3. Save assistant response to session state
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer_text,
        "sources": sources
    })
