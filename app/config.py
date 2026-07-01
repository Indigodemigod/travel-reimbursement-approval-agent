from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_ROOT_DIR = Path(__file__).resolve().parent.parent
_APP_DIR = Path(__file__).resolve().parent

load_dotenv(_ROOT_DIR / ".env")
load_dotenv(_APP_DIR / ".env", override=True)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(_ROOT_DIR / ".env", _APP_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    google_api_key: str = Field(
        validation_alias="GOOGLE_API_KEY",
        description="Google API key for Gemini",
    )

    @field_validator("google_api_key")
    @classmethod
    def validate_google_api_key(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("GOOGLE_API_KEY is required and must not be empty")
        return stripped


settings = Settings()
