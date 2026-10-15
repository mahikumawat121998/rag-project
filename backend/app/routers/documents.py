import os
import tempfile
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from backend.app.schemas.chat import DocumentUploadResponse, DocumentListResponse, DocumentItem
from backend.app.rag.ingest import ingest_uploaded_file, list_indexed_documents
from backend.app.rag.vectorstore import get_collection_count

logger = logging.getLogger("rag_api")
router = APIRouter(prefix="/documents", tags=["Documents Management"])

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".md"}

@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    summary="Upload & Index Document",
    description="Upload a PDF, DOCX, or TXT file to parse, chunk, embed, and store into ChromaDB in real-time."
)
async def upload_document(file: UploadFile = File(...)):
    # 1. Validate file extension
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Allowed formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    # 2. Save stream to temporary file
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
            file_size = len(content)

        logger.info(f"Uploading document: '{file.filename}' ({file_size} bytes)")

        # 3. Ingest into ChromaDB
        chunks_added, total_vectors = ingest_uploaded_file(temp_path, file.filename)

        # 4. Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

        return DocumentUploadResponse(
            filename=file.filename,
            file_size_bytes=file_size,
            chunks_added=chunks_added,
            total_vectors=total_vectors,
            message=f"Successfully indexed {chunks_added} chunks into ChromaDB!",
            success=True
        )
    except Exception as e:
        logger.error(f"Error indexing uploaded document '{file.filename}': {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to index document: {str(e)}"
        )

@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List All Indexed Documents",
    description="List all unique documents and their chunk counts stored in ChromaDB."
)
async def list_documents():
    try:
        docs = list_indexed_documents()
        total_chunks = sum(d["chunk_count"] for d in docs)
        return DocumentListResponse(
            documents=[DocumentItem(**d) for d in docs],
            total_documents=len(docs),
            total_chunks=total_chunks
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve document list: {str(e)}"
        )
