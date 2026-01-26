"""MAFA Configuration - YAML-based with environment overrides.

All settings are in config.yaml. Environment variables can override.
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


def _load_yaml_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    import yaml

    if not os.path.exists(config_path):
        return {}

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"Warning: Failed to load config.yaml: {e}")
        return {}


def _get_env_override(key: str, default: Any = None) -> Any:
    """Get value from environment variable, with prefix."""
    # Try MAFA_ prefix first
    env_key = f"MAFA_{key.upper()}"
    value = os.getenv(env_key)
    if value is not None:
        return value
    return default


def _parse_list(value: str) -> List[str]:
    """Parse comma-separated string to list."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return []


@dataclass
class NvidiaConfig:
    """NVIDIA NIM API configuration."""

    api_key: str = ""
    model: str = "llama-3.1-8b-instant"
    temperature: float = 0.1
    max_tokens: int = 512
    retry_attempts: int = 3
    retry_delay: int = 5


@dataclass
class HuggingFaceConfig:
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    use_local: bool = True


@dataclass
class FAISSConfig:
    dimension: int = 384
    metric: str = "cosine"


@dataclass
class AgentWeights:
    primary_only: float = 0.25
    contextual: float = 0.25
    retrieval: float = 0.25
    hybrid: float = 0.25


@dataclass
class RetrievalConfig:
    k_examples: int = 3


@dataclass
class ConfidenceThresholds:
    approve: float = 0.85
    review: float = 0.60


@dataclass
class JudgeConfig:
    timeout_ms: int = 200


@dataclass
class LoggingConfig:
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


@dataclass
class TaskColumns:
    text: str = "Comment"
    title: str = "Title"
    label: str = "Toxicity"


@dataclass
class TaskPaths:
    seed_file: str = ""


@dataclass
class TaskConsensus:
    approve_threshold: float = 0.85
    escalate_threshold: float = 0.60


@dataclass
class TaskConfig:
    type: str = "classification"
    description: str = "Phân loại toxicity cho comment tiếng Việt"
    labels: List[str] = field(default_factory=lambda: ["0", "1"])
    columns: TaskColumns = field(default_factory=TaskColumns)
    paths: TaskPaths = field(default_factory=TaskPaths)
    consensus: TaskConsensus = field(default_factory=TaskConsensus)

    def get_labels(self) -> List[str]:
        return self.labels


@dataclass
class NvidiaConfig:
    """NVIDIA NIM API configuration (loaded from config.yaml)."""

    api_key: str = ""
    model: str = ""  # Loaded from config.yaml
    base_url: str = ""
    temperature: float = 0.1
    max_tokens: int = 512
    enabled: bool = False


@dataclass
class ProviderConfig:
    """LLM provider configuration (NVIDIA)."""

    type: str = "nvidia"  # Default to nvidia


@dataclass
class Config:
    """Main configuration container."""

    provider: ProviderConfig
    nvidia: NvidiaConfig
    huggingface: HuggingFaceConfig
    faiss: FAISSConfig
    agents: AgentWeights
    retrieval: RetrievalConfig
    thresholds: ConfidenceThresholds
    judge: JudgeConfig
    logging: LoggingConfig
    task: TaskConfig


def load_config(config_path: str = "config.yaml") -> Config:
    """Load configuration from YAML with environment overrides."""
    yaml_config = _load_yaml_config(config_path)

    # Helper to get with env override
    def get(path: str, default: Any = None) -> Any:
        # Check env override
        env_value = _get_env_override(path)
        if env_value is not None:
            return env_value
        # Navigate YAML config
        keys = path.split(".")
        value = yaml_config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return default
        return value if value is not None else default

    # Helper to get nested config
    def get_section(path: str, default: Dict = {}) -> Dict:
        keys = path.split(".")
        value = yaml_config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key, {})
            else:
                return default
        return value if value else default

    return Config(
        provider=ProviderConfig(
            type=get("provider.type", "nvidia"),
        ),
        nvidia=NvidiaConfig(
            api_key=get("nvidia.api_key", "")
            or os.getenv("NVIDIA_API_KEY", "")
            or os.getenv("NIM_API_KEY", ""),
            model=get("nvidia.model", ""),  # Must be set in config.yaml
            base_url=get("nvidia.base_url", ""),
            temperature=float(get("nvidia.temperature", 0.1)),
            max_tokens=int(get("nvidia.max_tokens", 512)),
            enabled=get("nvidia.enabled", False),
        ),
        huggingface=HuggingFaceConfig(
            embedding_model=get(
                "embedding.model", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            use_local=get("embedding.use_local", True),
        ),
        faiss=FAISSConfig(
            dimension=int(get("faiss.dimension", 384)),
            metric=get("faiss.metric", "cosine"),
        ),
        agents=AgentWeights(
            primary_only=float(get("agents.weights.primary_only", 0.25)),
            contextual=float(get("agents.weights.contextual", 0.25)),
            retrieval=float(get("agents.weights.retrieval", 0.25)),
            hybrid=float(get("agents.weights.hybrid", 0.25)),
        ),
        retrieval=RetrievalConfig(
            k_examples=int(get("agents.retrieval.k_examples", 3)),
        ),
        thresholds=ConfidenceThresholds(
            approve=float(get("thresholds.approve", 0.85)),
            review=float(get("thresholds.review", 0.60)),
        ),
        judge=JudgeConfig(
            timeout_ms=int(get("judge.timeout_ms", 200)),
        ),
        logging=LoggingConfig(
            level=get("logging.level", "INFO"),
            format=get(
                "logging.format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            ),
        ),
        task=TaskConfig(
            type=get("task.type", "classification"),
            description=get(
                "task.description", "Phân loại toxicity cho comment tiếng Việt"
            ),
            labels=_parse_list(get("task.labels", ["0", "1"])),
            columns=TaskColumns(
                text=get("task.columns.text", "Comment"),
                title=get("task.columns.title", "Title"),
                label=get("task.columns.label", "Toxicity"),
            ),
            paths=TaskPaths(
                seed_file=get("task.paths.seed_file", ""),
            ),
            consensus=TaskConsensus(
                approve_threshold=float(get("task.consensus.approve_threshold", 0.85)),
                escalate_threshold=float(
                    get("task.consensus.escalate_threshold", 0.60)
                ),
            ),
        ),
    )


_config: Optional[Config] = None


def get_config(config_path: str = "config.yaml") -> Config:
    """Get or create config singleton."""
    global _config
    if _config is None:
        _config = load_config(config_path)
    return _config


def reset_config():
    """Reset config (for testing)."""
    global _config
    _config = None


def get_llm_client(config: Config = None):
    """Get LLM client based on provider configuration.

    Returns:
        NimClient instance, or None if not available
    """
    if config is None:
        config = get_config()

    provider_type = config.provider.type.lower()

    if provider_type == "nvidia" or config.nvidia.enabled:
        try:
            from .api.nim_client import NimClient

            return NimClient(
                model=config.nvidia.model,
                temperature=config.nvidia.temperature,
                max_tokens=config.nvidia.max_tokens,
                base_url=config.nvidia.base_url if config.nvidia.base_url else None,
                api_key=config.nvidia.api_key if config.nvidia.api_key else None,
            )
        except Exception as e:
            print(f"Warning: Could not initialize NIM client: {e}")
            return None

    print(f"Warning: Unknown provider '{provider_type}' or provider disabled.")
    return None
