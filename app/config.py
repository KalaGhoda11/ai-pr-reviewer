"""Application configuration loaded from environment variables.

Uses pydantic-settings so config is validated once at startup and injected
everywhere else, keeping secrets out of the code.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Gemini
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"

    # GitHub
    github_webhook_secret: str = ""
    github_token: str = ""

    # App
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (read env once per process)."""
    return Settings()
