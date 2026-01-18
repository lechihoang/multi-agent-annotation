"""
Configuration management using Pydantic Settings.

Loads configuration from environment variables and .env file.
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class APISettings(BaseSettings):
    """API keys and endpoints configuration."""

    groq_api_key: str = Field(default="", description="Groq API key")
    hf_token: str = Field(default="", description="HuggingFace token")
    ollama_host: str = Field(
        default="http://localhost:11434", description="Ollama server URL"
    )


class ModelSettings(BaseSettings):
    """Default model configurations."""

    default_groq_model: str = Field(
        default="deepseek-r1-distill-llama-70b",
        description="Default Groq model",
    )
    default_hf_model: str = Field(
        default="meta-llama/Llama-3.2-8B-Instruct",
        description="Default HuggingFace model",
    )
    default_ollama_model: str = Field(
        default="llama3.2",
        description="Default Ollama model",
    )


class ThresholdSettings(BaseSettings):
    """Consensus threshold configurations."""

    auto_approve_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Threshold for auto-approval",
    )
    human_review_threshold: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
        description="Threshold for human review",
    )


class AgentWeightSettings(BaseSettings):
    """Agent weight configurations for consensus voting."""

    intent_agent_weight: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Weight for intent classification agent",
    )
    entity_agent_weight: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Weight for entity extraction agent",
    )
    faq_agent_weight: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        description="Weight for FAQ matching agent",
    )


class QueueSettings(BaseSettings):
    """Queue configuration."""

    max_queue_size: int = Field(
        default=1000,
        ge=1,
        description="Maximum review queue size",
    )
    review_batch_size: int = Field(
        default=10,
        ge=1,
        description="Batch size for review processing",
    )


class ServerSettings(BaseSettings):
    """Server configuration."""

    backend_host: str = Field(default="0.0.0.0", description="Backend host")
    backend_port: int = Field(default=8000, description="Backend port")
    debug: bool = Field(default=False, description="Debug mode")


class LoggingSettings(BaseSettings):
    """Logging and monitoring configuration."""

    log_level: str = Field(default="INFO", description="Log level")
    audit_enabled: bool = Field(default=True, description="Enable audit logging")


class DatabaseSettings(BaseSettings):
    """Database configuration."""

    database_url: str = Field(
        default="sqlite+aiosqlite:///./data/annotations.db",
        description="Database connection URL",
    )


class Settings(BaseSettings):
    """Main settings class combining all configurations."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # API Settings
    groq_api_key: str = Field(default="")
    hf_token: str = Field(default="")
    ollama_host: str = Field(default="http://localhost:11434")

    # Model Settings
    default_groq_model: str = Field(default="deepseek-r1-distill-llama-70b")
    default_hf_model: str = Field(default="meta-llama/Llama-3.2-8B-Instruct")
    default_ollama_model: str = Field(default="llama3.2")

    # Threshold Settings
    auto_approve_threshold: float = Field(default=0.85)
    human_review_threshold: float = Field(default=0.60)

    # Agent Weights
    intent_agent_weight: float = Field(default=0.35)
    entity_agent_weight: float = Field(default=0.35)
    faq_agent_weight: float = Field(default=0.30)

    # Queue Settings
    max_queue_size: int = Field(default=1000)
    review_batch_size: int = Field(default=10)

    # Server Settings
    backend_host: str = Field(default="0.0.0.0")
    backend_port: int = Field(default=8000)
    debug: bool = Field(default=False)

    # Logging
    log_level: str = Field(default="INFO")
    audit_enabled: bool = Field(default=True)

    # Database
    database_url: str = Field(default="sqlite+aiosqlite:///./data/annotations.db")

    # Config file paths
    config_dir: Path = Field(
        default=Path(__file__).parent,
        description="Directory containing config files",
    )

    @property
    def models_yaml_path(self) -> Path:
        """Path to models.yaml configuration file."""
        return self.config_dir / "models.yaml"

    @property
    def prompts_yaml_path(self) -> Path:
        """Path to prompts.yaml configuration file."""
        return self.config_dir / "prompts.yaml"

    @property
    def agent_weights(self) -> dict[str, float]:
        """Get agent weights as a dictionary."""
        return {
            "intent_agent": self.intent_agent_weight,
            "entity_agent": self.entity_agent_weight,
            "faq_agent": self.faq_agent_weight,
        }

    @property
    def thresholds(self) -> dict[str, float]:
        """Get thresholds as a dictionary."""
        return {
            "auto_approve": self.auto_approve_threshold,
            "human_review": self.human_review_threshold,
            "escalate": 0.0,
        }


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Convenience function for accessing settings
settings = get_settings()
