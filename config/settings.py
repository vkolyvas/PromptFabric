from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""

    # LM Studio Configuration
    lm_studio_url: str = "http://localhost:1234/v1/chat/completions"

    # Model Configuration
    refiner_model: str = "gemma-3-4b-it"
    generator_model: str = "deepseek-coder-r1-7b"
    validator_model: str = "phi-3-mini-128k"

    # Vector Database
    vector_db_type: str = "chroma"  # chroma or qdrant
    chroma_persist_directory: str = "./chroma_data"
    qdrant_url: str = "http://localhost:6333"

    # SQLite Memory
    sqlite_db_path: str = "./memory.db"

    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Prompt Templates
    refiner_system_prompt: str = """You are a prompt refinement expert.
Your task is to convert user prompts into optimized, structured prompts
that will produce better results from an LLM.
Include relevant context, formatting, and structure."""

    generator_system_prompt: str = """You are a helpful AI assistant.
Provide accurate, well-structured responses based on the given context."""

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
