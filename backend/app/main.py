import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from backend.app.config import settings
from backend.app.routers.chat import router as chat_router
from backend.app.routers.documents import router as documents_router
from backend.app.rag.vectorstore import get_collection_count

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("rag_api")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Verify ChromaDB vector store
    logger.info("Initializing Nimbus Labs RAG Service...")
    count = get_collection_count()
    logger.info(f"Connected to ChromaDB: {count} vectors available in '{settings.COLLECTION_NAME}'")
    yield
    # Shutdown
    logger.info("Shutting down Nimbus Labs RAG Service.")

# Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=(
        "Production-ready RAG API powered by Google Gemini and ChromaDB.\n\n"
        "Ask questions about company leave, expenses, attendance, IT security, and NimbusDesk product documentation."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Enable CORS for frontend applications (Streamlit, React, Vue, Next.js)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(chat_router, prefix=settings.API_V1_PREFIX)
app.include_router(documents_router, prefix=settings.API_V1_PREFIX)

@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to interactive Swagger UI documentation."""
    return RedirectResponse(url="/docs")

@app.get("/health", tags=["Health"])
async def health_check():
    """Simple health check endpoint."""
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "database": "connected"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="127.0.0.1", port=8000, reload=True)
