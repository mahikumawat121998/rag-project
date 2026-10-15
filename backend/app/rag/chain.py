import re
from typing import List, Dict, Any, Tuple
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from backend.app.config import settings
from backend.app.rag.vectorstore import get_retriever
from backend.app.schemas.chat import SourceDocument

_llm = None

def get_llm() -> ChatGoogleGenerativeAI:
    """Returns the Gemini Chat model instance."""
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(
            model=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            google_api_key=settings.GOOGLE_API_KEY
        )
    return _llm

GREETING_PATTERNS = [
    r"^(hi|hello|hey|greetings|good\s+(morning|afternoon|evening|day)|howdy|sup)\b[!\.\?\s]*$"
]

def is_greeting(text: str) -> bool:
    cleaned = text.strip().lower()
    return any(re.match(p, cleaned) for p in GREETING_PATTERNS)

def reformulate_question(question: str, chat_history: List[Dict[str, str]]) -> str:
    """If chat history exists, reformulate follow-up questions (like 'Which of those...') into standalone queries."""
    if not chat_history:
        return question

    recent = chat_history[-4:]
    history_lines = []
    for turn in recent:
        role = turn.get("role", "user").capitalize()
        history_lines.append(f"{role}: {turn.get('content', '')}")
    history_text = "\n".join(history_lines)

    reformulate_prompt = ChatPromptTemplate.from_template(
        "Given the chat history and the latest user question, rephrase the question to be a standalone search query.\n"
        "Do NOT answer the question. Only output the reformulated standalone query.\n\n"
        "Chat History:\n{history}\n\n"
        "Latest Question: {question}\n"
        "Standalone Question:"
    )

    llm = get_llm()
    chain = reformulate_prompt | llm | StrOutputParser()
    try:
        standalone = chain.invoke({"history": history_text, "question": question})
        return standalone.strip()
    except Exception:
        return question

def format_docs(docs) -> str:
    """Format retrieved documents into a clean context string."""
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", None)
        page_info = f" (Page {page + 1})" if page is not None else ""
        formatted.append(f"--- Document #{i}: {source}{page_info} ---\n{doc.page_content.strip()}")
    return "\n\n".join(formatted)

def build_qa_prompt() -> ChatPromptTemplate:
    """Constructs the system and user prompt for RAG QA."""
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

    user_template = (
        "Context from Company Documents:\n"
        "{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )

    return ChatPromptTemplate.from_messages([
        ("system", system_instruction),
        ("human", user_template)
    ])

def query_rag_pipeline(question: str, chat_history: List[Dict[str, str]] = None) -> Tuple[str, List[SourceDocument]]:
    """
    Executes the full RAG pipeline:
    1. Detects pure chit-chat / greetings to avoid redundant document searches.
    2. Reformulates follow-up queries using conversation history.
    3. Retrieves top-k matching documents from ChromaDB.
    4. Generates an accurate, grounded answer using Gemini.
    5. Returns answer and structured source metadata.
    """
    # 1. Handle pure greetings
    if is_greeting(question):
        return (
            "Hello! I am **NimbusDesk AI**, your internal company policy assistant. "
            "How can I help you today with leave policies, remote work, expenses, or IT security?",
            []
        )

    # 2. Reformulate query if follow-up
    standalone_query = question
    if chat_history:
        standalone_query = reformulate_question(question, chat_history)

    # 3. Retrieve relevant chunks
    retriever = get_retriever()
    docs = retriever.invoke(standalone_query)

    # 4. Format context
    context = format_docs(docs)

    # 5. Build LCEL chain
    llm = get_llm()
    prompt = build_qa_prompt()
    chain = prompt | llm | StrOutputParser()

    # 6. Generate answer
    answer = chain.invoke({
        "context": context,
        "question": question
    })

    # 7. Extract structured source documents
    sources: List[SourceDocument] = []
    for doc in docs:
        meta = doc.metadata or {}
        raw_source = meta.get("source", "Unknown")
        clean_source = raw_source.replace("\\", "/").split("/")[-1]
        page = meta.get("page", None)

        sources.append(SourceDocument(
            source=clean_source,
            page=page + 1 if page is not None else None,
            content_preview=doc.page_content.strip()[:180] + "...",
            metadata=meta
        ))

    return answer, sources
