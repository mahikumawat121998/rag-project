import logging
from typing import List
from fastapi import APIRouter, HTTPException, status
from backend.app.schemas.chat import ChatRequest, ChatResponse, StatusResponse, SourceDocument
from backend.app.rag.chain import query_rag_pipeline
from backend.app.rag.vectorstore import get_collection_count, get_retriever
from backend.app.config import settings

logger = logging.getLogger("rag_api")
router = APIRouter(prefix="/chat", tags=["Chat & RAG"])

@router.post(
    "",
    response_model=ChatResponse,
    summary="Query Company Policy RAG",
    description="Ask a question against Nimbus Labs company policies and documents. Returns AI answer and cited sources."
)
async def chat_endpoint(request: ChatRequest):
    try:
        logger.info(f"Received query: '{request.question}'")
        answer, sources = query_rag_pipeline(
            question=request.question,
            chat_history=request.chat_history
        )
        return ChatResponse(
            question=request.question,
            answer=answer,
            sources=sources,
            success=True
        )
    except Exception as e:
        logger.error(f"Error executing RAG pipeline: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process question: {str(e)}"
        )

@router.get(
    "/status",
    response_model=StatusResponse,
    summary="Vector Store Status",
    description="Inspect ChromaDB status and total indexed vector count."
)
async def status_endpoint():
    try:
        count = get_collection_count()
        return StatusResponse(
            status="ready" if count > 0 else "empty",
            collection_name=settings.COLLECTION_NAME,
            total_vectors=count,
            persist_directory=settings.CHROMA_PERSIST_DIR
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch vector store status: {str(e)}"
        )
