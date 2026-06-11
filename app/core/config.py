from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    CACHE_TTL_SECONDS: int = 60 * 60 * 24  # 24h

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


class LLMSettings(BaseSettings):
    LLM_API_KEY: str
    LLM_API_BASE_URL: str
    LLM_API_BASE_MODEL: str

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


@lru_cache()
def get_settings() -> tuple[Settings, LLMSettings]:
    return Settings(), LLMSettings()


settings, llm_settings = get_settings()
