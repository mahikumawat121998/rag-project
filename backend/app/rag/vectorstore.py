import os
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from backend.app.config import settings

_embeddings = None
_vectorstore = None

def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Singleton getter for Google Gemini embeddings."""
    global _embeddings
    if _embeddings is None:
        _embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            google_api_key=settings.GOOGLE_API_KEY
        )
    return _embeddings

def get_vectorstore() -> Chroma:
    """Returns the persistent Chroma vector store instance."""
    global _vectorstore
    if _vectorstore is None:
        emb = get_embeddings()
        _vectorstore = Chroma(
            persist_directory=settings.CHROMA_PERSIST_DIR,
            embedding_function=emb,
            collection_name=settings.COLLECTION_NAME
        )
    return _vectorstore

def get_retriever(k: int = None):
    """Returns a retriever from Chroma vector store."""
    top_k = k or settings.RETRIEVER_K
    vs = get_vectorstore()
    return vs.as_retriever(search_kwargs={"k": top_k})

def get_collection_count() -> int:
    """Returns the total number of indexed vectors."""
    try:
        vs = get_vectorstore()
        return vs._collection.count()
    except Exception:
        return 0
