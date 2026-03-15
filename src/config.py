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


@dataclass
class NvidiaConfig:
    model: str = "meta/llama-3.3-70b-instruct"
    base_url: str = "https://integrate.api.nvidia.com/v1"
    temperature: float = 0.0  # DREAM uses temperature=0.0
    top_p: float = 0.7
    max_tokens: int = 2048
    stream: bool = False
    max_retries: int = 5
    rate_limit: int = 40
    api_key: str = field(
        default_factory=lambda: (
            os.getenv("NVIDIA_API_KEY", "") or os.getenv("NIM_API_KEY", "")
        )
    )


@dataclass
class TaskConfig:
    name: str = "complaint_detection"
    text_column: str = "review"
    label_column: str = "label"
    labels: Dict[str, str] = field(
        default_factory=lambda: {
            "0": "Non-complaint — Pure praise, satisfaction, OR insult/hate speech WITHOUT constructive intent.",
            "1": "Complaint — Dissatisfaction, suggestion, wish, warning, OR mixed WITH constructive intent.",
        }
    )


# ---------------------------------------------------------------------------
# DREAM Configuration (Multi-Agent Debate)
# Based on arXiv:2602.06526
# ---------------------------------------------------------------------------

@dataclass
class DreamDebateConfig:
    max_rounds: int = 2  # Paper: R=2 is optimal
    num_agents: int = 2  # 2 agents with opposing stances
    temperature: float = 0.0  # Deterministic output


@dataclass
class DreamAmbiguityConfig:
    enabled: bool = False  # DREAM uses agreement, not ambiguity detection
    detection_threshold: float = 0.5


@dataclass
class DreamLLMConfig:
    temperature: float = 0.0  # DREAM uses deterministic output
    max_tokens: int = 1024


@dataclass
class DreamEscalationConfig:
    enabled: bool = True  # Enable human escalation for disagreements
    use_llm_adjudicator: bool = True  # Use LLM as adjudicator instead of human
    adjudicator_model: Optional[str] = None  # Use same model as agents if null


@dataclass
class DreamConfig:
    enabled: bool = True
    guidelines: str = ""
    agent_complaint_system: str = ""
    agent_non_complaint_system: str = ""
    adjudicator_system: str = ""
    debate_round1_system: str = ""
    debate_roundN_system: str = ""
    debate: DreamDebateConfig = field(default_factory=DreamDebateConfig)
    ambiguity: DreamAmbiguityConfig = field(default_factory=DreamAmbiguityConfig)
    llm: DreamLLMConfig = field(default_factory=DreamLLMConfig)
    escalation: DreamEscalationConfig = field(default_factory=DreamEscalationConfig)


# ---------------------------------------------------------------------------
# Legacy MADISSE Configuration (backward compatibility)
# ---------------------------------------------------------------------------

@dataclass
class MadisseDebateConfig:
    max_rounds: int = 2
    num_agents: int = 2


@dataclass
class MadisseAmbiguityConfig:
    enabled: bool = True
    detection_threshold: float = 0.5


@dataclass
class MadisseLLMConfig:
    temperature: float = 0.6
    max_tokens: int = 2048


@dataclass
class MadisseConfig:
    enabled: bool = False  # Disabled by default, use DREAM instead
    guidelines: str = ""
    agent_complaint_system: str = ""
    agent_non_complaint_system: str = ""
    adjudicator_system: str = ""
    ambiguity_system: str = ""
    debate: MadisseDebateConfig = field(default_factory=MadisseDebateConfig)
    ambiguity: MadisseAmbiguityConfig = field(default_factory=MadisseAmbiguityConfig)
    llm: MadisseLLMConfig = field(default_factory=MadisseLLMConfig)


@dataclass
class Config:
    nvidia: NvidiaConfig
    task: TaskConfig
    dream: DreamConfig
    madisse: MadisseConfig  # Legacy, kept for backward compatibility
    logging_level: str = "INFO"


def load_config(path: str = "config.yaml") -> Config:
    d = _load_yaml(path)
    nv = _get(d, "nvidia") or {}
    task = _get(d, "task") or {}

    # Load DREAM config
    dream = _get(d, "dream") or {}
    dream_debate = _get(dream, "debate") or {}
    dream_ambig = _get(dream, "ambiguity") or {}
    dream_llm = _get(dream, "llm") or {}
    dream_escalation = _get(dream, "escalation") or {}

    # Load MADISSE config (legacy)
    mad = _get(d, "madisse") or {}
    mad_debate = _get(mad, "debate") or {}
    mad_ambig = _get(mad, "ambiguity") or {}
    mad_llm = _get(mad, "llm") or {}

    return Config(
        nvidia=NvidiaConfig(
            model=nv.get("model", "meta/llama-3.3-70b-instruct"),
            base_url=nv.get("base_url", "https://integrate.api.nvidia.com/v1"),
            temperature=float(nv.get("temperature", 0.0)),
            top_p=float(nv.get("top_p", 0.7)),
            max_tokens=int(nv.get("max_tokens", 2048)),
            stream=bool(nv.get("stream", False)),
            max_retries=int(nv.get("max_retries", 5)),
            rate_limit=int(nv.get("rate_limit", 40)),
        ),
        task=TaskConfig(
            name=task.get("name", "complaint_detection"),
            text_column=task.get("text_column", "review"),
            label_column=task.get("label_column", "label"),
            labels=task.get("labels", {"0": "Non-complaint", "1": "Complaint"}),
        ),
        dream=DreamConfig(
            enabled=dream.get("enabled", True),
            guidelines=dream.get("guidelines", ""),
            agent_complaint_system=dream.get("agent_complaint_system", ""),
            agent_non_complaint_system=dream.get("agent_non_complaint_system", ""),
            adjudicator_system=dream.get("adjudicator_system", ""),
            debate_round1_system=dream.get("debate_round1_system", ""),
            debate_roundN_system=dream.get("debate_roundN_system", ""),
            debate=DreamDebateConfig(
                max_rounds=int(dream_debate.get("max_rounds", 2)),
                num_agents=int(dream_debate.get("num_agents", 2)),
                temperature=float(dream_debate.get("temperature", 0.0)),
            ),
            ambiguity=DreamAmbiguityConfig(
                enabled=dream_ambig.get("enabled", False),
                detection_threshold=float(dream_ambig.get("detection_threshold", 0.5)),
            ),
            llm=DreamLLMConfig(
                temperature=float(dream_llm.get("temperature", 0.0)),
                max_tokens=int(dream_llm.get("max_tokens", 1024)),
            ),
            escalation=DreamEscalationConfig(
                enabled=dream_escalation.get("enabled", True),
                use_llm_adjudicator=dream_escalation.get("use_llm_adjudicator", True),
                adjudicator_model=dream_escalation.get("adjudicator_model"),
            ),
        ),
        madisse=MadisseConfig(
            enabled=mad.get("enabled", False),
            guidelines=mad.get("guidelines", ""),
            agent_complaint_system=mad.get("agent_complaint_system", ""),
            agent_non_complaint_system=mad.get("agent_non_complaint_system", ""),
            adjudicator_system=mad.get("adjudicator_system", ""),
            ambiguity_system=mad.get("ambiguity_system", ""),
            debate=MadisseDebateConfig(
                max_rounds=int(mad_debate.get("max_rounds", 2)),
                num_agents=int(mad_debate.get("num_agents", 2)),
            ),
            ambiguity=MadisseAmbiguityConfig(
                enabled=mad_ambig.get("enabled", True),
                detection_threshold=float(mad_ambig.get("detection_threshold", 0.5)),
            ),
            llm=MadisseLLMConfig(
                temperature=float(mad_llm.get("temperature", 0.6)),
                max_tokens=int(mad_llm.get("max_tokens", 2048)),
            ),
        ),
        logging_level=_get(d, "logging", "level") or "INFO",
    )


_config: Optional[Config] = None


def get_config(path: str = "config.yaml") -> Config:
    global _config
    if _config is None:
        _config = load_config(path)
    return _config


def reset_config():
    global _config
    _config = None
