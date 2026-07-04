"""
Config loader — YAML → dataclass for DREAM pipeline.
Task-agnostic: swap config.yaml to run on different NLP classification tasks.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


def _load_yaml(path: str = "config.yaml") -> Dict[str, Any]:
    import yaml

    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _get(d: Dict, *keys, default=None):
    v: Any = d
    for k in keys:
        if not isinstance(v, dict):
            return default
        v = v.get(k)
        if v is None:
            return default
    return v


# ---------------------------------------------------------------------------
# LLM / API
# ---------------------------------------------------------------------------

@dataclass
class NvidiaConfig:
    model: str = "meta/llama-3.3-70b-instruct"
    base_url: str = "https://integrate.api.nvidia.com/v1"
    max_tokens: int = 2048
    max_retries: int = 5
    rate_limit: int = 40


# ---------------------------------------------------------------------------
# Task Definition (task-agnostic)
# ---------------------------------------------------------------------------

@dataclass
class TaskConfig:
    name: str = "binary_classification"
    text_column: str = "review"
    label_column: str = "label"
    labels: Dict[str, str] = field(
        default_factory=lambda: {
            "0": "Label 0",
            "1": "Label 1",
        }
    )


# ---------------------------------------------------------------------------
# DREAM Pipeline
# ---------------------------------------------------------------------------

@dataclass
class DebateConfig:
    max_rounds: int = 2


@dataclass
class ModeratorConfig:
    enabled: bool = True
    system: str = ""
    user_template: str = ""


@dataclass
class AdjudicatorConfig:
    temperature: float = 0.0
    max_tokens: int = 1024
    system: str = ""
    user_template: str = ""


@dataclass
class EscalationConfig:
    mode: str = "fully_automatic"
    export_file: str = "data/escalated.csv"


@dataclass
class DreamLLMConfig:
    temperature: float = 0.0
    max_tokens: int = 2048


@dataclass
class DreamConfig:
    guidelines: str = ""
    agent_a_system: str = ""
    agent_b_system: str = ""
    agent_user_template: str = ""
    agent_roundN_template: str = ""
    moderator: ModeratorConfig = field(default_factory=ModeratorConfig)
    adjudicator: AdjudicatorConfig = field(default_factory=AdjudicatorConfig)
    escalation: EscalationConfig = field(default_factory=EscalationConfig)
    debate: DebateConfig = field(default_factory=DebateConfig)
    llm: DreamLLMConfig = field(default_factory=DreamLLMConfig)


@dataclass
class Config:
    nvidia: NvidiaConfig
    task: TaskConfig
    dream: DreamConfig
    logging_level: str = "INFO"


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_config(path: str = "config.yaml") -> Config:
    d = _load_yaml(path)

    nv = _get(d, "nvidia") or {}
    task = _get(d, "task") or {}
    dream = _get(d, "dream") or {}
    dream_debate = _get(dream, "debate") or {}
    mod = _get(dream, "moderator") or {}
    adj = _get(dream, "adjudicator") or {}
    esc = _get(dream, "escalation") or {}
    llm = _get(dream, "llm") or {}

    return Config(
        nvidia=NvidiaConfig(
            model=nv.get("model", "meta/llama-3.3-70b-instruct"),
            base_url=nv.get("base_url", "https://integrate.api.nvidia.com/v1"),
            max_tokens=int(nv.get("max_tokens", 2048)),
            max_retries=int(nv.get("max_retries", 5)),
            rate_limit=int(nv.get("rate_limit", 40)),
        ),
        task=TaskConfig(
            name=task.get("name", "binary_classification"),
            text_column=task.get("text_column", "review"),
            label_column=task.get("label_column", "label"),
            labels=task.get("labels", {"0": "Label 0", "1": "Label 1"}),
        ),
        dream=DreamConfig(
            guidelines=dream.get("guidelines", ""),
            agent_a_system=dream.get("agent_a_system", ""),
            agent_b_system=dream.get("agent_b_system", ""),
            agent_user_template=dream.get("agent_user_template", ""),
            agent_roundN_template=dream.get("agent_roundN_template", ""),
            moderator=ModeratorConfig(
                enabled=mod.get("enabled", True),
                system=mod.get("system", ""),
                user_template=mod.get("user_template", ""),
            ),
            adjudicator=AdjudicatorConfig(
                temperature=float(adj.get("temperature", 0.0)),
                max_tokens=int(adj.get("max_tokens", 1024)),
                system=adj.get("system", ""),
                user_template=adj.get("user_template", ""),
            ),
            escalation=EscalationConfig(
                mode=esc.get("mode", "fully_automatic"),
                export_file=esc.get("export_file", "data/escalated.csv"),
            ),
            debate=DebateConfig(
                max_rounds=int(dream_debate.get("max_rounds", 2)),
            ),
            llm=DreamLLMConfig(
                temperature=float(llm.get("temperature", 0.0)),
                max_tokens=int(llm.get("max_tokens", 2048)),
            ),
        ),
        logging_level=_get(d, "logging", "level") or "INFO",
    )


# ---------------------------------------------------------------------------
# Singleton cache
# ---------------------------------------------------------------------------

_config: Optional[Config] = None


def get_config(path: str = "config.yaml") -> Config:
    global _config
    if _config is None:
        _config = load_config(path)
    return _config


def reset_config():
    global _config
    _config = None
