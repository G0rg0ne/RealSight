"""Application settings loaded from environment variables."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the inference API."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    triton_url: str = "triton:8000"
    model_name: str = "catboost_model"
    model_version: str = "1"
    model_repository_path: str = "/models"
    model_cbm_path: str | None = None

    def resolved_model_cbm_path(self) -> str:
        """Path to the CatBoost model file (env override or default repository layout)."""
        if self.model_cbm_path:
            return self.model_cbm_path
        return (
            f"{self.model_repository_path.rstrip('/')}/"
            f"{self.model_name}/{self.model_version}/model.cbm"
        )


@lru_cache
def get_settings() -> Settings:
    """Cached settings instance."""
    return Settings()
