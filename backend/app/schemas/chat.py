from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class SourceDocument(BaseModel):
    """Metadata and snippet of a retrieved source chunk."""
    source: str = Field(description="Source file name or path")
    page: Optional[int] = Field(default=None, description="Page number if applicable")
    content_preview: str = Field(description="Preview of the retrieved chunk content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Full metadata attributes")

class ChatRequest(BaseModel):
    """User question payload."""
    question: str = Field(..., min_length=2, description="The user question for the policy chatbot")
    chat_history: Optional[List[Dict[str, str]]] = Field(
        default=[],
        description="Optional prior chat history as list of {'role': 'user'|'assistant', 'content': '...'}"
    )

class ChatResponse(BaseModel):
    """Chatbot answer payload with cited sources."""
    question: str
    answer: str
    sources: List[SourceDocument] = []
    success: bool = True

class StatusResponse(BaseModel):
    """ChromaDB and application status."""
    status: str
    collection_name: str
    total_vectors: int
    persist_directory: str

class DocumentUploadResponse(BaseModel):
    """Result of uploading and indexing a document."""
    filename: str
    file_size_bytes: int
    chunks_added: int
    total_vectors: int
    message: str
    success: bool = True

class DocumentItem(BaseModel):
    filename: str
    chunk_count: int

class DocumentListResponse(BaseModel):
    documents: List[DocumentItem]
    total_documents: int
    total_chunks: int

