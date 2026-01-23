# Multi-Agent Data Annotation System (MAFA)

MAFA-inspired multi-agent framework for automated data annotation with human-in-the-loop quality control.

## Overview

A production-ready data annotation system inspired by [MAFA: A Multi-Agent Framework for Enterprise-Scale Annotation](https://arxiv.org/abs/2510.14184). Uses 4 parallel agents with weighted consensus voting, supporting both Groq API and NVIDIA NIM API.

### Key Features

- **Dual LLM Support**: Groq API (free tier) or NVIDIA NIM API (production)
- **Batch Processing**: Process 10 samples per API call (vs 1 sample = 1 call)
- **Rate Limiting**: Configurable calls/minute to avoid rate limits
- **Low RAM**: ~500MB (local embeddings only, no model training)
- **Parallel Execution**: 4 annotation agents per batch
- **Weighted Consensus**: MAFA-style voting (0.25 each)
- **Human-in-the-Loop**: Review queue for low-confidence annotations

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ TIER 1: Query Planning                                          │
│ • Batch expansion (1 call = 10 samples)                         │
│ • QueryPlanner for query expansion                              │
└─────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│ TIER 2: Parallel Annotation Agents (4 calls per batch)         │
├─────────────────────────────────────────────────────────────────┤
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Agent 1: PrimaryOnly (LLM)                                │  │
│  │ • Direct classification without embeddings                 │  │
│  │ • Batch: 10 samples in 1 API call                         │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Agent 2: Contextual (LLM)                                 │  │
│  │ • LLM + secondary context                                 │  │
│  │ • Batch: 10 samples in 1 API call                         │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Agent 3: Retrieval (Embeddings + FAISS)                   │  │
│  │ • all-MiniLM-L6-v2 embeddings                             │  │
│  │ • Batch: 10 samples in 1 API call                         │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Agent 4: Hybrid (Embeddings + Unique Examples)            │  │
│  │ • Retrieval-MRL with edge cases                           │  │
│  │ • Batch: 10 samples in 1 API call                         │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│ TIER 3: Judge Agent (Consensus)                                 │
│ • Batch consensus (1 call = 10 samples)                         │
│ • Weighted voting: Score = Σ(conf × weight) / 4                 │
│ • Thresholds: approve (≥0.85), review (0.60-0.85), escalate    │
└─────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│ TIER 4: Human-in-the-Loop                                       │
│ • Auto-approved annotations                                     │
│ • Human review queue                                            │
│ • Expert escalation for low-confidence cases                    │
└─────────────────────────────────────────────────────────────────┘
```

## API Call Comparison

| Samples | Sequential (Original) | Batch (Optimized) |
|---------|----------------------|-------------------|
| 10 | 60 calls | 6 calls |
| 20 | 120 calls | 12 calls |
| 100 | 600 calls | 60 calls |
| 1000 | 6000 calls | 600 calls |
| 4000 | 24,000 calls | 2,400 calls |

**Time estimate (40 calls/minute):**
- 100 samples: ~2 minutes
- 1000 samples: ~15 minutes
- 4000 samples: ~60 minutes

## Installation

### Prerequisites

- Python 3.10+
- 512MB RAM minimum (8GB recommended)
- API key (Groq or NVIDIA NIM)

### Setup

```bash
cd multi-agent-annotation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Configure API Key

Edit `.env` file:

```bash
# Option 1: Groq API (FREE tier)
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxx

# Option 2: NVIDIA NIM API (Production)
NIM_API_KEY=nvapi-xxxxxxxxxxxxxxxxxxxxxxxx
```

## Configuration

Edit `config.yaml`:

```yaml
# Provider: "groq" or "nvidia"
provider:
  type: "nvidia"

# NVIDIA NIM settings
nvidia:
  enabled: true
  model: "meta/llama-3.3-70b-instruct"
  temperature: 0.1
  max_tokens: 1024

# Groq settings (fallback)
groq:
  model: "llama-3.3-70b-versatile"
  temperature: 0.1
  max_tokens: 1024

# Batch settings
batch_size: 10  # Samples per API call

# Rate limiting (calls per minute)
rate_limit: 40
```

## Usage

### Quick Test (5 samples)

```bash
./venv/bin/python scripts/run_batch.py --max 5 --batch-size 5 --rate 40
```

### Process 20 Samples (2 batches)

```bash
./venv/bin/python scripts/run_batch.py --max 20 --batch-size 10 --rate 40
```

### Process All Data

```bash
# 4000 samples = ~60 minutes at 40 calls/minute
./venv/bin/python scripts/run_batch.py --input data/train.csv --batch-size 10 --rate 40
```

### Custom Options

```bash
./venv/bin/python scripts/run_batch.py \
  --input data/train.csv \
  --output results.json \
  --batch-size 10 \
  --max 1000 \
  --rate 80  # Faster processing
```

### Human Review (Tier 4)

```bash
# Show statistics
./venv/bin/python scripts/review.py --stats

# Review items needing human input
./venv/bin/python scripts/review.py --input data/batch_mafa_results.json

# Auto-decide based on majority vote
./venv/bin/python scripts/review.py --input data/batch_mafa_results.json --auto

# Approve all review items
./venv/bin/python scripts/review.py --input data/batch_mafa_results.json --approve-all
```

### Process 20 Samples (2 batches)

```bash
./venv/bin/python run_batch_mafa.py --max 20 --batch-size 10 --rate 40
```

### Process All Data

```bash
# 4000 samples = ~60 minutes at 40 calls/minute
./venv/bin/python run_batch_mafa.py --input data/train.csv --batch-size 10 --rate 40
```

### Custom Options

```bash
./venv/bin/python run_batch_mafa.py \
  --input data/train.csv \
  --output results.json \
  --batch-size 10 \
  --max 1000 \
  --rate 80  # Faster processing
```

## API Providers

### Groq API (Free Tier)

- **Website**: https://console.groq.com/
- **Model**: llama-3.3-70b-versatile
- **Rate Limit**: Higher free tier limits
- **Setup**: `GROQ_API_KEY` in `.env`

### NVIDIA NIM API (Production)

- **Website**: https://build.nvidia.com/
- **Models**:
  - `meta/llama-3.3-70b-instruct`
  - `meta/llama-3.1-70b-instruct`
  - `deepseek-ai/deepseek-r1` (reasoning model)
- **Setup**: `NIM_API_KEY` in `.env`
- **Recommended**: Use instruct models for JSON output

## Batch Processing Flow

```
Input CSV (train.csv)
       │
       ▼
┌──────────────────┐
│ Batch: 10 texts  │
└────────┬─────────┘
         │
         ▼
┌─────────────────────────────────────┐
│ TIER 1: Query Expansion (1 call)    │
│ → "comment text" → expanded query   │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│ TIER 2: 4 Agents (4 calls)          │
│ → PrimaryOnly: batch 10 samples     │
│ → Contextual: batch 10 samples      │
│ → Retrieval: batch 10 samples       │
│ → Hybrid: batch 10 samples          │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│ TIER 3: Judge Consensus (1 call)    │
│ → Combine 4 agent results           │
│ → Calculate final confidence        │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│ TIER 4: Review Queue                │
│ → approve (≥0.85)                   │
│ → review (0.60-0.85)                │
│ → escalate (<0.60)                  │
└────────┬────────────────────────────┘
         │
         ▼
Output JSON (results.json)
```

## Project Structure

```
multi-agent-annotation/
├── scripts/
│   ├── run_batch.py      # Main batch processing script
│   └── review.py         # Tier 4: Human review CLI
├── src/
│   ├── api/
│   │   ├── groq_client.py     # Groq API client
│   │   └── nim_client.py      # NVIDIA NIM API client
│   ├── agents/
│   │   ├── tier1/             # Router, QueryPlanner, QueryExpander
│   │   ├── tier2/             # 4 annotation agents
│   │   ├── tier3/             # Judge consensus
│   │   └── tier4/             # Human review workflow
│   ├── config.py        # Centralized configuration
│   ├── main.py          # Original AnnotationPipeline
│   ├── consensus/       # Voting algorithms
│   └── monitoring/      # Metrics & weight updates
├── data/
│   ├── train.csv        # Training data
│   └── batch_mafa_results.json  # Output
├── config.yaml
├── .env
├── requirements.txt
└── README.md
## Output Format

```json
{
  "text": "Comment text here",
  "expanded": "Expanded query",
  "tier2": {
    "primary": {"label": "0", "confidence": 0.95},
    "contextual": {"label": "0", "confidence": 0.92},
    "retrieval": {"label": "0", "confidence": 0.88},
    "hybrid": {"label": "0", "confidence": 0.90}
  },
  "tier3": {
    "label": "0",
    "confidence": 0.91,
    "decision": "approve"
  },
  "batch": 1,
  "task_id": "batch1_0"
}
```

## Decision Thresholds

| Confidence | Decision | Action |
|------------|----------|--------|
| ≥ 0.85 | approve | Auto-approved |
| 0.60 - 0.85 | review | Human review |
| < 0.60 | escalate | Expert escalation |

## Dependencies

```
groq>=0.1.0              # Cloud LLM inference
openai>=1.0.0            # NVIDIA NIM API (OpenAI-compatible)
httpx>=0.27.0            # Async HTTP client
sentence-transformers>=2.2.0  # Embeddings
faiss-cpu>=1.7.0         # Vector similarity search
python-dotenv>=1.0.0     # Environment config
numpy>=1.24.0            # Numerical operations
pandas>=2.0.0            # Data handling
```

## References

- [MAFA: A Multi-Agent Framework for Enterprise-Scale Annotation](https://arxiv.org/abs/2510.14184) - JP Morgan
- [Groq API](https://console.groq.com/)
- [NVIDIA NIM](https://build.nvidia.com/)
- [sentence-transformers](https://www.sbert.net/)

## License

MIT
