from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Scientific Search API"
    app_version: str = "0.1.0"
    app_env: str = "development"
    log_level: str = "INFO"

    api_host: str = "0.0.0.0"
    api_port: int = 8000

    vector_store_dir: str = Field(default="data/vector_store", validation_alias="SCISEARCH_VECTOR_STORE_DIR")
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    request_timeout_seconds: int = 30
    answer_max_substeps_default: int = 4
    answer_top_k_per_step_default: int = 3

    langfuse_enabled: bool = False
    langfuse_host: str = "http://langfuse:3000"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_environment: str = "local"


def get_settings() -> Settings:
    return Settings()
