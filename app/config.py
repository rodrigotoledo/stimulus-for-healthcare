"""Runtime configuration, loaded from environment / .env with the HFD_ prefix."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="HFD_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- HuggingFace -------------------------------------------------------
    hf_hub_url: str = "https://huggingface.co"
    hf_datasets_server_url: str = "https://datasets-server.huggingface.co"
    hf_token: str | None = None
    hf_timeout: float = 30.0

    # --- Ollama ------------------------------------------------------------
    ollama_url: str = "http://localhost:11434"
    ollama_default_model: str = "MedGemma:4b"
    ollama_connect_timeout: float = 5.0
    ollama_read_timeout: float = 600.0

    # --- HTTP / paging -----------------------------------------------------
    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ]
    )
    max_page_size: int = 50
    default_page_size: int = 10

    @property
    def hf_headers(self) -> dict[str, str]:
        headers = {"User-Agent": "hf-dialog-selector/0.2"}
        if self.hf_token:
            headers["Authorization"] = f"Bearer {self.hf_token}"
        return headers


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
