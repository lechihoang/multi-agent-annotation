# Multi-Agent Data Annotation System

## Overview

A production-ready multi-agent data annotation system using free AI APIs (Groq, HuggingFace, Ollama) with CrewAI framework. Achieves 70-80% automation rate while maintaining 85%+ quality standards through tiered architecture and consensus mechanisms.

## Architecture

### Tiered Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Tier 1: Router Agent                                        │
│ • Task complexity classification                            │
│ • API routing (Groq ↔ HF ↔ Ollama)                         │
│ • Confidence thresholding                                   │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ Tier 2: Specialized Annotation Agents (Parallel)            │
│ • Intent Classification Agent (DeepSeek-R1, Groq)           │
│ • Entity Extraction Agent (Llama 70B, Groq/HF)              │
│ • FAQ/Pattern Matching Agent (Llama 8B, HF/Ollama)          │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ Tier 3: Consensus & Quality Control                         │
│ • Multi-agent weighted voting                               │
│ • Confidence score aggregation                              │
│ • Low-confidence flagging                                   │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ Tier 4: Output & Human-in-the-Loop                          │
│ • High confidence (>0.85): Auto-approve                     │
│ • Medium (0.60-0.85): Human review queue                    │
│ • Low (<0.60): Expert escalation                            │
└─────────────────────────────────────────────────────────────┘
```

### Framework Selection

| Framework | Use Case | Why |
|-----------|----------|-----|
| **CrewAI** | Primary orchestration | Simple 4 primitives (Agents, Tasks, Tools, Crew), production-ready, role-based design |
| **LangGraph** | Complex branching | Graph-based state, checkpointing, human interrupts |
| **AutoGen** | Conversation patterns | Message-passing, built-in error handling |

**Current Choice**: CrewAI for simplicity and production reliability.

### API Strategy

#### Free Tier Allocation

| Provider | Monthly Allocation | Primary Use | Fallback |
|----------|-------------------|-------------|----------|
| **Groq** | Generous credits | Complex reasoning, entity extraction | HuggingFace |
| **HuggingFace** | Monthly credits | Pattern matching, experimentation | Ollama (local) |
| **Ollama** | Unlimited (local) | Validation, preprocessing, simple tasks | N/A |

#### Model Selection

| Task Type | Primary Model | Provider | Complexity |
|-----------|---------------|----------|------------|
| Intent Classification | DeepSeek-R1 | Groq | High |
| Entity Extraction | Llama 70B | Groq/HF | Medium |
| FAQ Matching | Llama 8B | HF Free | Low |
| Validation | Llama 7B | Ollama (local) | Simple |

### Consensus Mechanism

#### Weighted Voting Algorithm

```
Score = Σ(agent_confidence × agent_weight) / N

Agent Weights:
- Intent Agent: 0.35 (critical for downstream)
- Entity Agent: 0.35 (core extraction)
- FAQ Agent: 0.30 (pattern matching)
```

#### Confidence Thresholds

| Score Range | Action | Target Rate |
|-------------|--------|-------------|
| > 0.85 | Auto-approve | 60-70% |
| 0.60 - 0.85 | Human review | 20-30% |
| < 0.60 | Expert escalation | 5-10% |

## Project Structure

```
multi-agent-annotation/
├── src/
│   ├── agents/
│   │   ├── router_agent.py         # Tier 1: Task classification & routing
│   │   ├── intent_agent.py         # Tier 2: Intent classification
│   │   ├── entity_agent.py         # Tier 2: Entity extraction
│   │   ├── faq_agent.py            # Tier 2: FAQ/pattern matching
│   │   └── judge_agent.py          # Tier 3: Consensus & quality
│   │
│   ├── api/
│   │   ├── groq_client.py          # Groq API integration
│   │   ├── huggingface_client.py   # HuggingFace integration
│   │   └── ollama_client.py        # Ollama local inference
│   │
│   ├── consensus/
│   │   ├── voting.py               # Weighted voting algorithm
│   │   ├── confidence.py           # Confidence aggregation
│   │   └── thresholds.py           # Threshold rules
│   │
│   ├── human_review/
│   │   ├── queue.py                # Review queue management
│   │   ├── prioritization.py       # Priority scoring
│   │   └── workflow.py             # Review workflow
│   │
│   ├── monitoring/
│   │   ├── metrics.py              # Performance metrics
│   │   ├── logging.py              # Structured logging
│   │   └── observability.py        # Tracing & audit
│   │
│   └── config/
│       ├── models.yaml             # Model configurations
│       ├── prompts.yaml            # Agent prompts
│       └── settings.py             # Environment settings
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
│
├── data/
│   ├── input/                      # Raw data for annotation
│   ├── output/                     # Annotated results
│   └── cache/                      # Cached API responses
│
├── docs/
│   ├── architecture.md
│   ├── api.md
│   └── runbook.md
│
├── scripts/
│   ├── setup.sh                    # Environment setup
│   ├── run_pipeline.py             # Main pipeline
│   └── monitor.sh                  # Monitoring dashboard
│
├── requirements.txt
├── .env.example
└── CLAUDE.md                       # This file
```

## Implementation Guidelines

### Agent Development

#### 1. Router Agent (Tier 1)

```python
# src/agents/router_agent.py

class RouterAgent:
    """
    Tier 1: Analyzes task complexity and routes to appropriate API/model.
    
    Responsibilities:
    - Classify task complexity (high/medium/low)
    - Route to appropriate API provider
    - Apply confidence thresholds
    - Batch similar tasks for efficiency
    """
    
    def classify_complexity(self, task: Task) -> ComplexityLevel:
        """Determine task complexity based on content analysis."""
        
    def route_to_api(self, task: Task, complexity: ComplexityLevel) -> APIResponse:
        """Route task to optimal API provider."""
        
    def apply_thresholds(self, result: AnnotationResult) -> RoutingDecision:
        """Apply confidence thresholds for routing decisions."""
```

#### 2. Specialized Agents (Tier 2)

```python
# src/agents/intent_agent.py

class IntentAgent:
    """
    Tier 2: Classify user intents using DeepSeek-R1 (Groq).
    
    Model: DeepSeek-R1
    Provider: Groq (high-complexity routing)
    Weight: 0.35
    """
    
    def __init__(self):
        self.client = GroqClient()
        self.model = "deepseek-r1-distill-llama-70b-spec"
        self.weight = 0.35
        
    async def classify_intent(self, text: str) -> IntentAnnotation:
        """Classify intent with confidence score."""
        
    def get_prompt(self, text: str) -> str:
        """Generate intent classification prompt."""
```

```python
# src/agents/entity_agent.py

class EntityAgent:
    """
    Tier 2: Extract named entities using Llama 70B.
    
    Model: Llama 70B
    Provider: Groq or HuggingFace
    Weight: 0.35
    """
    
    def __init__(self, provider: str = "groq"):
        self.client = GroqClient() if provider == "groq" else HFClient()
        self.model = "llama-3.3-70b-versatile"  # or hf equivalent
        self.weight = 0.35
        
    async def extract_entities(self, text: str) -> EntityAnnotation:
        """Extract entities with confidence score."""
```

```python
# src/agents/faq_agent.py

class FAQAgent:
    """
    Tier 2: Match against known patterns and FAQs.
    
    Model: Llama 8B
    Provider: HuggingFace (free tier) or Ollama (local)
    Weight: 0.30
    """
    
    def __init__(self, provider: str = "hf"):
        self.client = HFClient() if provider == "hf" else OllamaClient()
        self.model = "meta-llama/Llama-3.2-8B-Instruct"  # or smaller
        self.weight = 0.30
        
    async def match_faq(self, text: str) -> FAQAnnotation:
        """Match text against FAQ patterns."""
```

#### 3. Judge Agent (Tier 3)

```python
# src/agents/judge_agent.py

class JudgeAgent:
    """
    Tier 3: Consensus mechanism and quality control.
    
    Responsibilities:
    - Aggregate multi-agent votes
    - Calculate weighted consensus scores
    - Flag low-confidence results
    - Generate audit trails
    """
    
    def calculate_consensus(
        self, 
        annotations: List[Annotation]
    ) -> ConsensusResult:
        """Calculate weighted consensus score."""
        
    def flag_for_review(self, score: float, annotations: List[Annotation]) -> ReviewFlag:
        """Flag low-confidence results for human review."""
        
    def generate_audit_trail(
        self, 
        task: Task, 
        annotations: List[Annotation],
        decision: Decision
    ) -> AuditTrail:
        """Generate traceable audit trail."""
```

### API Integration

#### Groq Client

```python
# src/api/groq_client.py

from groq import Groq
from typing import List, Dict
import os

class GroqClient:
    """
    Groq API integration for high-speed inference.
    
    Models:
    - deepseek-r1-distill-llama-70b-spec (reasoning)
    - llama-3.3-70b-versatile (general)
    """
    
    def __init__(self):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        
    async def chat_completion(
        self, 
        model: str, 
        messages: List[Dict[str, str]],
        temperature: float = 0.1
    ) -> ChatResponse:
        """Execute chat completion with Groq."""
        
    async def structured_output(
        self, 
        model: str, 
        schema: Dict
    ) -> StructuredResponse:
        """Get structured output (JSON mode)."""
```

#### HuggingFace Client

```python
# src/api/huggingface_client.py

from huggingface_hub import InferenceClient
from typing import List, Dict
import os

class HuggingFaceClient:
    """
    HuggingFace Inference Providers integration.
    
    Supports:
    - 200+ models from various providers
    - OpenAI SDK compatibility
    - Free tier credits
    """
    
    def __init__(self):
        self.client = InferenceClient(
            base_url="https://inference-api.huggingface.com/v1",
            api_key=os.environ.get("HF_TOKEN")
        )
        
    async def chat_completion(
        self, 
        model: str, 
        messages: List[Dict[str, str]]
    ) -> ChatResponse:
        """Execute chat completion via HF Inference Providers."""
```

#### Ollama Client

```python
# src/api/ollama_client.py

import ollama
from typing import List, Dict

class OllamaClient:
    """
    Ollama local inference for zero-cost processing.
    
    Use cases:
    - Validation tasks
    - Preprocessing
    - Simple classification
    """
    
    def __init__(self, model: str = "llama3.2"):
        self.model = model
        
    async def chat_completion(
        self, 
        messages: List[Dict[str, str]]
    ) -> ChatResponse:
        """Execute local inference with Ollama."""
```

### Consensus Implementation

#### Voting Algorithm

```python
# src/consensus/voting.py

from typing import List
from dataclasses import dataclass

@dataclass
class Annotation:
    agent_name: str
    confidence: float
    result: Dict
    weight: float

@dataclass
class ConsensusResult:
    score: float
    decision: str  # "approve", "review", "escalate"
    votes: List[Dict]
    audit_trail: Dict

class ConsensusEngine:
    """
    Weighted voting consensus mechanism.
    
    Formula: Score = Σ(agent_confidence × agent_weight) / N
    """
    
    AGENT_WEIGHTS = {
        "intent_agent": 0.35,
        "entity_agent": 0.35,
        "faq_agent": 0.30
    }
    
    THRESHOLDS = {
        "approve": 0.85,
        "review": 0.60,
        "escalate": 0.0
    }
    
    def calculate_consensus(
        self, 
        annotations: List[Annotation]
    ) -> ConsensusResult:
        """Calculate weighted consensus score."""
        
        weighted_sum = sum(
            ann.confidence * ann.weight 
            for ann in annotations
        )
        
        score = weighted_sum / len(annotations)
        
        decision = self._get_decision(score)
        
        return ConsensusResult(
            score=score,
            decision=decision,
            votes=[self._serialize_vote(a) for a in annotations],
            audit_trail=self._generate_audit(annotations, score, decision)
        )
```

### Human-in-the-Loop

#### Review Queue

```python
# src/human_review/queue.py

from typing import List, Deque
from collections import deque
from dataclasses import dataclass, field
import heapq

@dataclass(order=True)
class ReviewItem:
    priority: int
    task: Task
    annotations: List[Annotation]
    consensus_score: float
    created_at: float
    
class ReviewQueue:
    """
    Priority queue for human review.
    
    Priority order:
    1. Escalated cases (score ≤ 0.60)
    2. Medium confidence (0.60 < score ≤ 0.85)
    3. Spot-check (random sampling)
    """
    
    def __init__(self):
        self.escalated: Deque = deque()
        self.medium: Deque = deque()
        self.spot_check: List = []
        
    def add(self, item: ReviewItem):
        """Add item to appropriate queue."""
        
    def get_next(self) -> ReviewItem:
        """Get next item for review (priority-based)."""
```

## Configuration

### Model Configuration

```yaml
# src/config/models.yaml

models:
  groq:
    intent_classification:
      model: "deepseek-r1-distill-llama-70b-spec"
      temperature: 0.1
      max_tokens: 1024
    entity_extraction:
      model: "llama-3.3-70b-versatile"
      temperature: 0.0
      max_tokens: 2048
      
  huggingface:
    faq_matching:
      model: "meta-llama/Llama-3.2-8B-Instruct"
      provider: "hf-inference"
      temperature: 0.1
      
  ollama:
    validation:
      model: "llama3.2"
      temperature: 0.0
      
agent_weights:
  intent_agent: 0.35
  entity_agent: 0.35
  faq_agent: 0.30
  
thresholds:
  auto_approve: 0.85
  human_review: 0.60
  expert_escalate: 0.0
```

### Environment Variables

```bash
# .env.example

# API Keys
GROQ_API_KEY=your_groq_api_key
HF_TOKEN=your_huggingface_token

# Model Settings
DEFAULT_GROQ_MODEL=deepseek-r1-distill-llama-70b-spec
DEFAULT_HF_MODEL=meta-llama/Llama-3.2-8B-Instruct
DEFAULT_OLLAMA_MODEL=llama3.2

# Queue Settings
MAX_QUEUE_SIZE=1000
REVIEW_BATCH_SIZE=10

# Logging
LOG_LEVEL=INFO
AUDIT_ENABLED=true
```

## Development Workflow

### Setup

```bash
# Clone and setup
git clone <repo>
cd multi-agent-annotation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Verify setup
python scripts/verify_setup.py
```

### Running the Pipeline

```bash
# Run full pipeline
python scripts/run_pipeline.py --input data/input/sample.json

# Run with specific config
python scripts/run_pipeline.py --config production --batch-size 100

# Run monitoring dashboard
python scripts/monitor.py --port 8000
```

### Testing

```bash
# Unit tests
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# End-to-end tests
pytest tests/e2e/ -v

# Generate coverage report
pytest --cov=src tests/ --cov-report=html
```

## Quality Standards

### Code Quality

- **Type Hints**: All functions must have type annotations
- **Documentation**: Docstrings for all public methods
- **Testing**: Minimum 80% unit test coverage
- **Linting**: Pass ruff/flake8 with zero warnings
- **Type Checking**: Pass mypy with strict mode

### Annotation Quality

| Metric | Target | Measurement |
|--------|--------|-------------|
| Agent Agreement | ≥85% | Consensus score |
| Automation Rate | 70-80% | Auto-approved / Total |
| Human Review Accuracy | ≥95% | Human agreement |
| Latency (p95) | <5s | Per annotation |

### Monitoring Metrics

```python
# src/monitoring/metrics.py

METRICS = {
    "annotation_count": "counter",
    "automation_rate": "gauge",
    "consensus_score_avg": "histogram",
    "human_review_queue_size": "gauge",
    "api_latency_p50": "histogram",
    "api_latency_p95": "histogram",
    "api_cost_current": "gauge",
    "agent_agreement_rate": "gauge",
    "error_count": "counter"
}
```

## Deployment

### Docker

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN chmod +x scripts/*.sh

CMD ["python", "scripts/run_pipeline.py"]
```

### Docker Compose

```yaml
# docker-compose.yml
services:
  annotation-service:
    build: .
    environment:
      - GROQ_API_KEY=${GROQ_API_KEY}
      - HF_TOKEN=${HF_TOKEN}
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    ports:
      - "8000:8000"
```

## Cost Optimization

### Free Tier Strategy

| Provider | Monthly Credits | Usage Strategy |
|----------|-----------------|----------------|
| Groq | Generous | Complex tasks (60%), Entity extraction (30%), Reserved (10%) |
| HF | Monthly | Pattern matching (40%), Experimentation (30%), Backup (30%) |
| Ollama | Unlimited | Validation, preprocessing, simple tasks |

### Optimization Techniques

1. **Smart Routing**: Route simple tasks to free/local providers
2. **Caching**: Cache FAQ matches and common patterns
3. **Batching**: Group similar tasks for batch processing
4. **Model Selection**: Use smallest capable model

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Rate limit errors | API quota exceeded | Switch to fallback provider |
| Low confidence scores | Ambiguous input | Increase human review threshold |
| High latency | Model overload | Add request queuing |
| Memory issues | Large batch sizes | Reduce batch size |

### Debug Commands

```bash
# Check API status
python scripts/debug_api_status.py

# View logs
tail -f logs/annotation.log

# Monitor queue
python scripts/monitor_queue.py --status

# Test consensus
python scripts/test_consensus.py --test-case sample
```

## Contributing

### Development Process

1. **Branch**: Create feature branch from `main`
2. **Develop**: Implement with tests and documentation
3. **Review**: Submit PR with detailed description
4. **Test**: CI/CD pipeline runs all tests
5. **Merge**: Approved by maintainers

### Code Standards

- Follow PEP 8 with ruff formatting
- Write docstrings in Google style
- Add type hints to all functions
- Maintain 80%+ test coverage
- Update documentation for new features

## Resources

### Documentation

- [CrewAI Documentation](https://docs.crewai.com/)
- [Groq API Docs](https://console.groq.com/docs/)
- [HuggingFace Inference](https://huggingface.co/docs/inference-providers/)
- [Ollama Documentation](https://ollama.com/)

### Papers & Research

- [MAFA: Multi-Agent Framework for Annotation](https://arxiv.org/html/2510.14184v1)
- [CrowdAgent: Multi-Source Annotation](https://arxiv.org/abs/2509.14030)
- [Transforming Data Annotation with AI Agents](https://www.mdpi.com/1999-5903/17/8/353)
