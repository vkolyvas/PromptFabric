from typing import Optional

from pydantic_settings import BaseSettings

from config.hardware_detect import (
    detect_hardware,
    get_recommended_models,
    get_recommended_provider,
)


def get_auto_settings():
    """Get automatically configured settings based on hardware."""
    hardware = detect_hardware()
    provider = get_recommended_provider(hardware)
    models = get_recommended_models(provider, hardware)

    class AutoSettings(BaseSettings):
        llm_provider: str = provider

        lm_studio_url: str = "http://localhost:1234/v1/chat/completions"
        ollama_url: str = "http://localhost:11434"
        ollama_model: str = models["generator"]

        refiner_model: str = models["refiner"]
        generator_model: str = models["generator"]
        validator_model: str = models["validator"]

        vector_db_type: str = "chroma"
        chroma_persist_directory: str = "./chroma_data"
        qdrant_url: str = "http://localhost:6333"

        sqlite_db_path: str = "./memory.db"

        api_host: str = "127.0.0.1"
        api_port: int = 8030

        refiner_system_prompt: str = """You are a prompt refinement expert.
Your task is to convert user prompts into optimized, structured prompts
that will produce better results from an LLM.
Include relevant context, formatting, and structure."""

        generator_system_prompt: str = """You are a helpful AI assistant.
Provide accurate, well-structured responses based on the given context."""

        class Config:
            env_file = ".env"
            env_file_encoding = "utf-8"

    return AutoSettings()


class Settings(BaseSettings):
    """Application settings"""

    # Auto-detect setting
    auto_detect: bool = True

    # LLM Provider: "lm_studio" or "ollama" or "auto"
    llm_provider: str = "auto"

    # LM Studio Configuration
    lm_studio_url: str = "http://localhost:1234/v1/chat/completions"

    # Ollama Configuration
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"

    # Model Configuration
    refiner_model: str = "gemma:2b"
    generator_model: str = "llama3.2:3b"
    validator_model: str = "phi3:3.8b"

    # Vector Database
    vector_db_type: str = "chroma"
    chroma_persist_directory: str = "./chroma_data"
    qdrant_url: str = "http://localhost:6333"

    # SQLite Memory
    sqlite_db_path: str = "./memory.db"

    # API Settings
    api_host: str = "127.0.0.1"
    api_port: int = 8030

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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Auto-detect hardware and apply recommendations
        if self.auto_detect and self.llm_provider == "auto":
            hardware = detect_hardware()
            self.llm_provider = get_recommended_provider(hardware)
            models = get_recommended_models(self.llm_provider, hardware)

            # Apply recommended models
            self.ollama_model = models["generator"]
            self.refiner_model = models["refiner"]
            self.generator_model = models["generator"]
            self.validator_model = models["validator"]

            # Override with env vars if explicitly set
            if "OLLAMA_MODEL" in kwargs:
                self.ollama_model = kwargs["OLLAMA_MODEL"]


# Global settings instance
settings = Settings()
