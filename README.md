# DREAM: Multi-Agent Debate Annotation for Vietnamese Complaint Detection

Implementation of **DREAM** (Debate-based RElevance Assessment with Multi-agents) from paper [arXiv:2602.06526](https://arxiv.org/abs/2602.06526).

## Overview

DREAM uses **multi-agent debate** with opposing stances to annotate Vietnamese e-commerce reviews (viOCD dataset).

### Key Features (from paper)

1. **Opposing Stance Initialization**: Two agents with opposing initial stances:
   - Agent_Relevant: Stance = "This is a COMPLAINT (Label 1)"
   - Agent_Irrelevant: Stance = "This is NOT a COMPLAINT (Label 0)"

2. **Multi-Round Debating with Reciprocal Critique**:
   - Agents debate for R rounds (default R=2)
   - Each agent critiques opponent's arguments
   - Evidence extraction from review text

3. **Agreement-based Human Escalation**:
   - If agents agree → use agreed label (high confidence)
   - If agents disagree → use LLM adjudicator (or escalate to human)

### Results (from paper)

| Method | Balanced Accuracy | Escalation Ratio |
|--------|------------------|------------------|
| DREAM (R=2) | **95.2%** | **3.5%** |
| LLMJudge (single agent) | 73.9% | 0.0% |
| LARA (confidence-based) | 82.1% | 12.5% |

## Installation

```bash
# Install dependencies
uv sync

# Or using pip
pip install -r requirements.txt
```

## Running on Kaggle

Create a new Notebook and run this in one cell:

```python
# Clone the project
!git clone https://github.com/your-repo/multi-agent-annotation.git
%cd multi-agent-annotation

# Set up NVIDIA API key
import os
os.environ["NVIDIA_API_KEY"] = "your-api-key-here"  # Or use Kaggle Secrets

# Install uv and dependencies
!pip install uv
!uv sync

# Run annotation
!uv run python run_annotation.py --input data/unlabeled.csv --output data/annotated.csv --concurrency 5

# Download results
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()
api.dataset_download_file('your-username/your-dataset', 'annotated.csv')
```

### Kaggle GPU/TPU Note

This project uses **NVIDIA NIM API** (cloud-based), so it doesn't require GPU on Kaggle. The API rate limit is 40 requests/minute, which is the main bottleneck.

## Configuration

Edit `config.yaml` to customize:

- LLM provider (NVIDIA NIM)
- Debate parameters (max_rounds, temperature)
- Label definitions for your task
- Agent prompts

## Usage

### 1. Annotate Data

```bash
# Annotate unlabeled reviews
uv run python run_annotation.py --input data/unlabeled.csv --output data/annotated.csv

# Limit to first N samples
uv run python run_annotation.py --input data/unlabeled.csv --output data/annotated.csv --limit 100

# With custom concurrency
uv run python run_annotation.py --input data/unlabeled.csv --output data/annotated.csv --concurrency 5
```

### 2. Relabel Existing Dataset

```bash
# Relabel training set (compares with old labels)
uv run python run_annotation.py --input data/train_labeled.csv --output data/train_relabeled.csv
```

## Output Format

CSV output with columns:

| Column | Description |
|--------|-------------|
| `task_id` | Unique task identifier |
| `review` | Original review text |
| `old_label` | Original label (for relabeling) |
| `final_label` | `0` (non-complaint) or `1` (complaint) |
| `confidence` | Confidence score (0.0-1.0) |
| `reasoning` | Reasoning for the decision |
| `reached_agreement` | `True` if agents agreed |
| `agreement_round` | Round where agreement was reached (1 or 2) |
| `used_adjudicator` | `True` if LLM adjudicator was used |
| `adjudication_reasoning` | Adjudicator's reasoning |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      DREAM Pipeline                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐     ┌──────────────┐                    │
│  │ Agent_Relevant│     │Agent_Irrelevant│                  │
│  │ (Stance: 1)  │     │  (Stance: 0)  │                  │
│  └──────┬───────┘     └──────┬───────┘                    │
│         │                     │                             │
│         └─────────┬───────────┘                             │
│                   ▼                                         │
│          ┌───────────────┐                                  │
│          │ Debate Round 1 │                                  │
│          │ (Reciprocal    │                                  │
│          │  Critique)     │                                  │
│          └───────┬────────┘                                  │
│                  │                                           │
│          ┌──────┴──────┐                                    │
│          │   Check     │                                    │
│          │ Agreement?  │                                    │
│          └──────┬──────┘                                    │
│                 │                                           │
│     ┌──────────┴──────────┐                                │
│     ▼                     ▼                                │
│  ┌──────┐           ┌─────────────┐                        │
│  │ YES  │           │    NO       │                        │
│  └──────┘           └──────┬──────┘                        │
│     │                      │                                │
│     ▼                      ▼                                │
│  ┌────────────┐     ┌─────────────┐                         │
│  │   Use      │     │ Debate R2   │                         │
│  │ agreed     │     │ or          │                         │
│  │ label      │     │ Adjudicator │                         │
│  └────────────┘     └─────────────┘                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Dataset

The viOCD (Vietnamese Online Complaint Detection) dataset:

- **Task**: Binary classification (complaint vs non-complaint)
- **Label 0**: Non-complaint - pure praise, satisfaction, or insult without constructive intent
- **Label 1**: Complaint - dissatisfaction, suggestion, wish, warning, or mixed with constructive intent

## Adaptation

To adapt for different tasks:

1. Modify `task.labels` in `config.yaml`
2. Update `dream.guidelines` with task-specific rules
3. Update agent prompts (`agent_complaint_system`, `agent_non_complaint_system`)

## License

MIT License
