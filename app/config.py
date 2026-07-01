from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    OLLAMA_BASE_URL:str = "http://localhost:11434"
    LLM_MODEL:str = "qwen3"
    EMBEDDING_MODEL:str = "bge-large"

    EMBEDDING_DIM:int = 1024

    CHUNK_SIZE:int = 1000
    CHUNK_OVERLAP:int = 200
    CHUNKING_STRATEGY:str = "semantic"

    TOP_K:int = 15
    RERANKING_TOP_K:int = 5
    SIMILARITY_THRESHOLD:float|None = None

    SUPPORTED_EXTENSIONS:list[str] = [".pdf"]

    EMBEDDING_CACHE_MAX_SIZE:int = 10000

    RETRIEVAL_CACHE_MAX_SIZE:int = 1000
    RETRIEVAL_CACHE_TTL_SECONDS:int = 3600

    LLM_CACHE_TTL_SECONDS:int = 3600
    LLM_CACHE_MAX_SIZE:int = 1000

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings() # type: ignore[call-arg]
