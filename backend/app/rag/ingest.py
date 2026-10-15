import os
import shutil
from typing import Tuple, List, Dict
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from backend.app.rag.vectorstore import get_vectorstore, get_collection_count
from backend.app.config import BASE_DIR, settings

UPLOAD_DIR = str(BASE_DIR / "data" / "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

def load_document_by_path(file_path: str):
    """Load PDF, DOCX, or TXT file based on file extension."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext in [".docx", ".doc"]:
        loader = Docx2txtLoader(file_path)
    elif ext in [".txt", ".md"]:
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {ext}. Allowed: .pdf, .docx, .txt")
    return loader.load()

def ingest_uploaded_file(temp_file_path: str, original_filename: str) -> Tuple[int, int]:
    """
    Ingests an uploaded document into ChromaDB:
    1. Saves copy in data/uploads
    2. Parses content with appropriate loader
    3. Splits into chunks preserving metadata
    4. Embeds and adds to persistent ChromaDB
    Returns (chunks_added, total_vectors_in_db)
    """
    # 1. Target persistent file path
    target_path = os.path.join(UPLOAD_DIR, original_filename)
    shutil.copy2(temp_file_path, target_path)

    # 2. Load document
    docs = load_document_by_path(target_path)
    if not docs:
        raise ValueError(f"No text content could be extracted from '{original_filename}'.")

    # Update metadata to reflect user-facing filename
    for doc in docs:
        doc.metadata["source"] = original_filename
        doc.metadata["uploaded"] = "true"

    # 3. Chunk documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(docs)

    # 4. Add into Chroma vector store
    vs = get_vectorstore()
    vs.add_documents(chunks)

    total_count = get_collection_count()
    return len(chunks), total_count

def list_indexed_documents() -> List[Dict[str, any]]:
    """Returns a summary of all unique documents stored in ChromaDB."""
    vs = get_vectorstore()
    try:
        all_records = vs._collection.get(include=["metadatas"])
        metadatas = all_records.get("metadatas", [])
        
        doc_stats: Dict[str, int] = {}
        for m in metadatas:
            raw_src = m.get("source", "Unknown")
            clean_name = raw_src.replace("\\", "/").split("/")[-1]
            doc_stats[clean_name] = doc_stats.get(clean_name, 0) + 1

        result = [
            {"filename": fname, "chunk_count": count}
            for fname, count in sorted(doc_stats.items())
        ]
        return result
    except Exception:
        return []
