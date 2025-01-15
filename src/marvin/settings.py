"""Settings for Marvin."""

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_ai.models import KnownModelName
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing_extensions import Self


class Settings(BaseSettings):
    """Settings for Marvin.

    Settings can be set via environment variables with the prefix MARVIN_. For
    example, MARVIN_AGENT_MODEL="openai:gpt-4o-mini"
    """

    model_config = SettingsConfigDict(
        env_prefix="MARVIN_",
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="forbid",
        validate_assignment=True,
    )

    # ------------ General settings ------------

    home_path: Path = Field(
        default=Path("~/.marvin"),
        description="The home path for Marvin.",
    )

    @field_validator("home_path")
    @classmethod
    def validate_home_path(cls, v: Path) -> Path:
        """Ensure the home path exists."""
        path = Path(v).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    database_path: Path | None = Field(
        default=None,
        description="Path to the database file. Defaults to `home_path / 'marvin.db'`.",
    )

    @model_validator(mode="after")
    def validate_database_path(self) -> Self:
        """Set and validate the database path."""
        # Skip validation for in-memory database
        if self.database_url == "sqlite+aiosqlite:///:memory:":
            return self

        # Extract path from database URL for file-based databases
        if self.database_url.startswith("sqlite"):
            # Get the path part after :///
            db_path = self.database_url.split("://", 1)[1]
            if not db_path.startswith("/"):
                # Relative path - store in home directory
                db_path = self.home_path / db_path
            else:
                db_path = Path(db_path)

            # Ensure parent directory exists
            db_path = db_path.expanduser().resolve()
            db_path.parent.mkdir(parents=True, exist_ok=True)

            # Update the URL with the resolved path
            self.database_url = f"sqlite+aiosqlite:///{db_path}"

        return self

    # ------------ Logging settings ------------

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level",
    )

    log_events: bool = Field(
        default=False,
        description="Whether to log all events (as debug logs).",
    )

    @field_validator("log_level", mode="before")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        """Validate the log level."""
        return v.upper()

    @model_validator(mode="after")
    def setup_logging(self) -> Self:
        """Finalize the settings."""
        import marvin.utilities.logging

        marvin.utilities.logging.setup_logging(self.log_level)

        return self

    # ------------ Agent settings ------------

    agent_model: KnownModelName = Field(
        default="openai:gpt-4o",
        description="The default model for agents.",
    )

    agent_temperature: float | None = Field(
        default=None,
        description="The temperature for the agent.",
    )

    agent_retries: int = Field(
        default=10,
        description="The number of times the agent is allowed to retry when it generates an invalid result.",
    )

    max_agent_turns: int | None = Field(
        default=100,
        description="The maximum number of turns any agents can take when running orchestrated tasks. Note this is per-invocation.",
    )

    # ------------ DX settings ------------

    enable_default_print_handler: bool = Field(
        default=True,
        description="Whether to enable the default print handler.",
    )

    # ------------ Memory settings ------------

    memory_provider: str = Field(
        default="chroma-db",
        description="The default memory provider for agents.",
    )

    chroma_cloud_api_key: str | None = Field(
        default=None,
        description="The API key for the Chroma Cloud.",
    )

    chroma_cloud_tenant: str | None = Field(
        default=None,
        description="The tenant for the Chroma Cloud.",
    )

    chroma_cloud_database: str | None = Field(
        default=None,
        description="The database for the Chroma Cloud.",
    )


# Global settings instance
settings = Settings()
