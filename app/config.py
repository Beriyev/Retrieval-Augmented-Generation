from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    primary_model: str = "qwen3"

    langchain_tracing_v2: bool = True 
    langchain_api_key: str = ""
    langchain_project: str = "production-api"

    app_env: str = "development"