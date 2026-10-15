import streamlit as st
import os
import re
import tempfile
from typing import List, Dict, Any

from dotenv import load_dotenv
load_dotenv(".env")

# Determine logo path reliably (supports root and subfolder execution)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(CURRENT_DIR, "assets", "logo.png")
if not os.path.exists(LOGO_PATH):
    LOGO_PATH = os.path.join(os.path.dirname(CURRENT_DIR), "assets", "logo.png")
APP_ICON = LOGO_PATH if os.path.exists(LOGO_PATH) else "🤖"

# Page configuration
st.set_page_config(
    page_title="NimbusDesk AI - Enterprise Policy RAG",
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- CONFIGURATION & SECRETS -----------------
# Safely resolve secrets from Streamlit Cloud (st.secrets) or local environment
def get_config(key: str, default: str = "") -> str:
    if key in st.secrets:
        return str(st.secrets[key])
    return os.getenv(key, default)

GOOGLE_API_KEY = get_config("GOOGLE_API_KEY")
SUPABASE_URL = get_config("SUPABASE_URL")
SUPABASE_KEY = get_config("SUPABASE_SERVICE_KEY") or get_config("SUPABASE_KEY")
API_BASE_URL = get_config("API_BASE_URL", "http://127.0.0.1:8000")

# Check which backend mode is active
USE_SUPABASE = bool(SUPABASE_URL and SUPABASE_KEY and "your-project-id" not in SUPABASE_URL)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #2563eb 0%, #7c3aed 50%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #64748b;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
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
    [data-testid="stSidebar"] img {
        border-radius: 16px;
        box-shadow: 0 8px 24px rgba(37, 99, 235, 0.25);
        margin-bottom: 0.75rem;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SUPABASE RAG ENGINE (CLOUD MODE) -----------------
@st.cache_resource
def get_supabase_vectorstore():
    if not USE_SUPABASE:
        return None
    from supabase.client import create_client
    from langchain_community.vectorstores import SupabaseVectorStore
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        output_dimensionality=768,
        google_api_key=GOOGLE_API_KEY
    )
    return SupabaseVectorStore(
        client=supabase_client,
        embedding=embeddings,
        table_name="documents",
        query_name="match_documents"
    )

def query_supabase_rag(question: str, chat_history: List[Dict[str, str]] = None):
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    # 1. Greeting check
    if re.match(r"^(hi|hello|hey|greetings|good\s+(morning|afternoon|evening|day)|howdy|sup)\b[!\.\?\s]*$", question.strip().lower()):
        return (
            "Hello! I am **NimbusDesk AI**, your internal company policy assistant. "
            "How can I help you today with leave policies, remote work, expenses, or IT security?",
            []
        )

    vs = get_supabase_vectorstore()
    if vs is None:
        return "Supabase vector store is not configured.", []

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        temperature=0.1,
        google_api_key=GOOGLE_API_KEY
    )

    # 2. History-aware query reformulation if needed
    standalone_query = question
    if chat_history and len(chat_history) > 0:
        recent = chat_history[-4:]
        history_text = "\n".join(f"{turn.get('role', 'user').capitalize()}: {turn.get('content', '')}" for turn in recent)
        reformulate_prompt = ChatPromptTemplate.from_template(
            "Given the chat history and the latest user question, rephrase the question to be a standalone search query.\n"
            "Do NOT answer the question. Only output the reformulated standalone query.\n\n"
            "Chat History:\n{history}\n\n"
            "Latest Question: {question}\n"
            "Standalone Question:"
        )
        try:
            standalone_query = (reformulate_prompt | llm | StrOutputParser()).invoke({
                "history": history_text,
                "question": question
            }).strip()
        except Exception:
            standalone_query = question

    # 3. Retrieve from Supabase
    docs = vs.similarity_search(standalone_query, k=5)

    # 4. Format context
    formatted_docs = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", None)
        page_info = f" (Page {page + 1})" if page is not None else ""
        formatted_docs.append(f"--- Document #{i}: {source}{page_info} ---\n{doc.page_content.strip()}")
    context = "\n\n".join(formatted_docs)

    # 5. QA Prompt
    system_instruction = (
        "You are NimbusDesk AI, the official company policy and documentation assistant at Nimbus Labs.\n"
        "Your task is to answer employee questions accurately and strictly based on the provided company policy context.\n\n"
        "Strict Guidelines:\n"
        "1. Strictly use the provided Context. Do NOT assume, extrapolate, or invent information not present.\n"
        "2. If the answer cannot be found in the context (or is not mentioned in the company documents), politely state: "
        "'I do not have enough information in the provided company documents to answer that question.'\n"
        "3. Provide clear, direct, well-structured, professional answers. Quote exact figures, limits, and roles when given.\n"
        "4. When multiple policies apply (e.g. IT security and remote work), synthesize information from all relevant documents cited."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_instruction),
        ("human", "Context from Company Documents:\n{context}\n\nQuestion: {question}\n\nAnswer:")
    ])

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})

    sources = []
    for doc in docs:
        meta = doc.metadata or {}
        raw_source = meta.get("source", "Unknown")
        clean_source = raw_source.replace("\\", "/").split("/")[-1]
        page = meta.get("page", None)
        sources.append({
            "source": clean_source,
            "page": page + 1 if page is not None else None,
            "content_preview": doc.page_content.strip()[:180] + "..."
        })

    return answer, sources

def ingest_to_supabase(uploaded_file):
    """Parses and ingests an uploaded file directly into Supabase pgvector."""
    from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    ext = os.path.splitext(uploaded_file.name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tf:
        tf.write(uploaded_file.getvalue())
        temp_path = tf.name

    try:
        if ext == ".pdf":
            loader = PyPDFLoader(temp_path)
        elif ext in [".docx", ".doc"]:
            loader = Docx2txtLoader(temp_path)
        else:
            loader = TextLoader(temp_path, encoding="utf-8")
        docs = loader.load()

        for d in docs:
            d.metadata["source"] = uploaded_file.name

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
        chunks = text_splitter.split_documents(docs)

        vs = get_supabase_vectorstore()
        vs.add_documents(chunks)
        return True, len(chunks)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# ----------------- LOCAL API FALLBACK -----------------
def query_local_api(question: str, chat_history: List[Dict[str, str]] = None):
    import requests
    try:
        payload = {"question": question, "chat_history": chat_history or []}
        res = requests.post(f"{API_BASE_URL}/api/chat", json=payload, timeout=60)
        if res.status_code == 200:
            data = res.json()
            return data.get("answer", ""), data.get("sources", [])
        return f"Backend Error: {res.text}", []
    except Exception as e:
        return f"Could not reach local FastAPI server ({API_BASE_URL}). If deploying to Streamlit Cloud, configure Supabase credentials in Streamlit Secrets.", []

# ----------------- SESSION STATE -----------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I am **NimbusDesk AI**, your enterprise policy and documentation assistant.\n\nAsk me anything about leave, remote work, expenses, IT security, or upload new policies!",
            "sources": []
        }
    ]

# ----------------- SIDEBAR -----------------
with st.sidebar:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=110)
    else:
        st.image("https://img.icons8.com/clouds/200/artificial-intelligence.png", width=90)
    st.title("Nimbus Hub")

    if USE_SUPABASE:
        st.success("☁️ Mode: Supabase pgvector (Cloud)", icon="⚡")
        st.caption(f"Connected to: `{SUPABASE_URL.split('//')[-1].split('.')[0]}`")
    else:
        st.info("🖥️ Mode: Local FastAPI / ChromaDB", icon="🔌")
        st.caption("To run 100% on cloud, add `SUPABASE_URL` to secrets.")

    st.divider()

    # Document Upload Section
    st.subheader("📤 Ingest New Document")
    uploaded_file = st.file_uploader(
        "Upload PDF, DOCX, or TXT",
        type=["pdf", "docx", "txt", "md"],
        help="Upload new policy documents to index into the vector store in real-time."
    )

    if uploaded_file is not None:
        if st.button("🚀 Upload & Ingest into Vector Store", use_container_width=True, type="primary"):
            with st.spinner(f"Ingesting '{uploaded_file.name}'..."):
                if USE_SUPABASE:
                    try:
                        ok, chunks_count = ingest_to_supabase(uploaded_file)
                        st.success(f"✅ Ingested {chunks_count} chunks into Supabase!")
                        st.toast(f"'{uploaded_file.name}' indexed successfully in Supabase!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to ingest: {e}")
                else:
                    import requests
                    try:
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type or "application/octet-stream")}
                        res = requests.post(f"{API_BASE_URL}/api/documents/upload", files=files, timeout=40)
                        if res.status_code == 200:
                            st.success(f"✅ Ingested {res.json().get('chunks_added', 0)} chunks!")
                            st.rerun()
                        else:
                            st.error(f"Upload failed: {res.text}")
                    except Exception as e:
                        st.error(f"Upload error: {e}")

    st.divider()

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
ASSISTANT_AVATAR = LOGO_PATH if os.path.exists(LOGO_PATH) else "🤖"
USER_AVATAR = "🧑‍💻"

header_col1, header_col2 = st.columns([1, 7], vertical_alignment="center")
with header_col1:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=82)
with header_col2:
    st.markdown('<div class="main-title">NimbusDesk AI Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Enterprise Conversational RAG with Google Gemini & Vector Storage</div>', unsafe_allow_html=True)

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
    with st.chat_message(msg["role"], avatar=USER_AVATAR if msg["role"] == "user" else ASSISTANT_AVATAR):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("🔍 Cited Sources & Page Numbers"):
                for idx, src in enumerate(msg["sources"], 1):
                    source_name = src.get("source", "Unknown")
                    page = src.get("page")
                    page_badge = f"<span class='source-badge'>Page {page}</span>" if page else ""
                    st.markdown(f"**Source #{idx}:** `{source_name}` {page_badge}", unsafe_allow_html=True)
                    st.caption(f"\"{src.get('content_preview', '')}\"")

# Handle input
user_query = st.chat_input("Ask a question about company policies...")
prompt_to_process = quick_prompt or user_query

if prompt_to_process:
    st.session_state.messages.append({"role": "user", "content": prompt_to_process, "sources": []})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt_to_process)

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        with st.spinner("Searching company documents and generating answer..."):
            prior_history = st.session_state.messages[:-1]
            if USE_SUPABASE:
                answer_text, sources = query_supabase_rag(prompt_to_process, chat_history=prior_history)
            else:
                answer_text, sources = query_local_api(prompt_to_process, chat_history=prior_history)

            st.markdown(answer_text)

            if sources:
                with st.expander("🔍 Cited Sources & Page Numbers"):
                    for idx, src in enumerate(sources, 1):
                        source_name = src.get("source", "Unknown")
                        page = src.get("page")
                        page_badge = f"<span class='source-badge'>Page {page}</span>" if page else ""
                        st.markdown(f"**Source #{idx}:** `{source_name}` {page_badge}", unsafe_allow_html=True)
                        st.caption(f"\"{src.get('content_preview', '')}\"")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer_text,
        "sources": sources
    })
