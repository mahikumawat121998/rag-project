import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# Base directory of project (one level above backend)
BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    # App info
    PROJECT_NAME: str = "Nimbus Labs Policy RAG API"
    VERSION: str = "1.0.0"
    API_V1_PREFIX: str = "/api"

    # Gemini settings
    GOOGLE_API_KEY: str = ""
    LLM_MODEL: str = "gemini-3.5-flash-lite"
    LLM_TEMPERATURE: float = 0.1
    EMBEDDING_MODEL: str = "models/gemini-embedding-001"

    # ChromaDB settings
    CHROMA_PERSIST_DIR: str = str(BASE_DIR / "data" / "chroma_db")
    COLLECTION_NAME: str = "gemini_company_policy"
    RETRIEVER_K: int = 5

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
