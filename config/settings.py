import os
from pathlib import Path

if 'SSL_CERT_FILE' in os.environ and not os.path.exists(os.environ['SSL_CERT_FILE']):
    del os.environ['SSL_CERT_FILE']

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    GEMINI_API_KEY: str | None = None
    # Override in .env to switch models. Defaults to flash-lite for the most
    # generous free-tier quota; bump to "gemini-2.0-flash" or "gemini-2.5-flash"
    # if you have billing enabled and want higher quality.
    GEMINI_MODEL: str = "gemini-2.5-flash"
    OPENAI_API_KEY: str | None = None
    ANTHROPIC_API_KEY: str | None = None

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

settings = Settings()
