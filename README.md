# Multi-Agent Data Annotation System (MAFA)

MAFA-inspired multi-agent framework for automated data annotation with human-in-the-loop quality control.
Production-ready implementation supporting **Dynamic Tasks** (e.g., Complaint Detection, Toxicity, Sentiment) via configuration.

## Overview

A robust data annotation system inspired by [MAFA: A Multi-Agent Framework for Enterprise-Scale Annotation](https://arxiv.org/abs/2510.14184). It leverages a 4-tier architecture with parallel agents and weighted consensus to achieve high-quality automated labeling.

### Key Features

- **Dynamic Task Configuration**: Switch between tasks (ViOCD, Toxicity, etc.) just by editing `config.yaml`.
- **Blind Annotation Support**: Strict separation of Seed data (for RAG learning) and Unlabeled data (for annotation).
- **LLM Support**: NVIDIA NIM API (Production) or OpenAI-compatible endpoints.
- **ARQ Reasoning**: Agents use "Attentive Reasoning Queries" (step-by-step thinking) for higher accuracy.
- **Robust Batch Processing**:
    - Resume capability (auto-skip processed rows).
    - CSV Output (Append mode).
    - Rate limiting protection.
- **Low Resource**: Runs efficiently on local machines (CPU-based FAISS + Cloud LLM).

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ TIER 1: Query Planning & Expansion                              │
│ • Analyzes intent and expands query context                     │
│ • Powered by LLM + Embedding Similarity                         │
└─────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│ TIER 2: Parallel Annotation Agents                              │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────┐ ┌─────────────────────────────┐│
│  │ Agent 1: PrimaryOnly        │ │ Agent 2: Contextual         ││
│  │ • Direct reasoning (ARQ)    │ │ • Considers domain/title    ││
│  └─────────────────────────────┘ └─────────────────────────────┘│
│                                                                 │
│  ┌─────────────────────────────┐ ┌─────────────────────────────┐│
│  │ Agent 3: Retrieval (RAG)    │ │ Agent 4: Hybrid (MRL)       ││
│  │ • Learns from Seed Data     │ │ • Focuses on Edge Cases     ││
│  └─────────────────────────────┘ └─────────────────────────────┘│
└─────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│ TIER 3: Judge Agent (Consensus)                                 │
│ • Weighted voting: Score = Σ(conf × weight) / 4                 │
│ • Dynamic weights based on historical accuracy                  │
│ • Configurable thresholds (Approve/Review/Escalate)             │
└─────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│ TIER 4: Human-in-the-Loop                                       │
│ • Review Queue for low-confidence items                         │
│ • Auto-approval for high-confidence items                       │
└─────────────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites
- Python 3.10+
- API Key (Groq or NVIDIA NIM)

### Setup

```bash
# Clone repository
git clone https://github.com/your-repo/multi-agent-annotation.git
cd multi-agent-annotation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Configure API Key
Create/Edit `.env` file:

```bash
# NVIDIA NIM API (or OpenAI-compatible)
NIM_API_KEY=nvapi-xxxxxxxxxxxxxxxxxxxxxxxx
```

## Configuration (Dynamic Tasks)

Edit `config.yaml` to define your annotation task.
Example for **ViOCD (Complaint Detection)**:

```yaml
task:
  name: "complaint_detection"
  description: "Phân loại phàn nàn (Complaint) trong bình luận thương mại điện tử tiếng Việt."
  
  # Define your labels here
  labels:
    "0": "Non-complaint - Bình luận tích cực, khen ngợi, hỏi đáp, hoặc trung tính."
    "1": "Complaint - Phàn nàn về sản phẩm, dịch vụ, giao hàng, đóng gói."
  
  # Map CSV columns
  columns:
    text: "review"   # Column containing text to annotate
    label: "label"   # Column containing ground truth (for seed data)
  
  # Consensus Thresholds
  consensus:
    approve_threshold: 0.85
    escalate_threshold: 0.60

  # Data Paths
  paths:
    seed_file: "data/seed.csv"  # File containing labeled examples for RAG
```

## Workflow: Blind Annotation

### 1. Prepare Data
Split your dataset into **Seed** (labeled examples for Agent learning) and **Unlabeled** (data to annotate).

```bash
# Automatically split train.csv into seed.csv (100 samples) and unlabeled.csv (rest)
# Also cleans unnecessary columns to ensure "blind" annotation.
./venv/bin/python scripts/prepare_data.py --train train.csv --dev dev.csv --seed 100
```

### 2. Run Annotation
Start the batch annotation process. The system allows stopping and resuming at any time.

```bash
# Run with rate limit (e.g., 30 calls/minute for Production)
./venv/bin/python scripts/run_arq_batch.py --input data/unlabeled.csv --rate 30
```

**Features:**
- **Auto-Resume**: Detects existing output file and skips processed rows.
- **CSV Output**: Saves results to `data/batch_arq_results.csv` immediately after each batch.
- **Safe**: No data loss if interrupted.

### 3. Review Results
Output format in `batch_arq_results.csv`:

| text | final_label | confidence | decision |
|------|-------------|------------|----------|
| "Sản phẩm tốt..." | 0 | 0.95 | approve |
| "Giao hàng chậm..." | 1 | 0.88 | approve |
| "Hàng cũng được..." | 0 | 0.65 | review |

## Advanced Usage

### Kaggle / Colab (Free Compute)
To run this system on Kaggle (ideal for long-running batches):

1. **Create a New Notebook** on Kaggle.
2. **Add Your API Key**: Go to "Add-ons" -> "Secrets" -> Add `NIM_API_KEY`.
3. **Upload Data**: Upload your `seed.csv` and `unlabeled.csv` as a Kaggle Dataset (or use the ones in the repo).
4. **Run the following code block**:

```python
# 1. Setup Environment
!git clone https://github.com/your-username/multi-agent-annotation.git
%cd multi-agent-annotation
!pip install -q -r requirements.txt

# 2. Load API Key
from kaggle_secrets import UserSecretsClient
import os
os.environ["NIM_API_KEY"] = UserSecretsClient().get_secret("NIM_API_KEY")

# 3. Prepare Data (Move from Input to Writable Directory)
# Assuming you uploaded data as a dataset named 'my-annotation-data'
!mkdir -p data
!cp /kaggle/input/my-annotation-data/seed.csv data/
!cp /kaggle/input/my-annotation-data/unlabeled.csv data/
# OR if using repo data, just skip this step

# 4. Run Annotation (Rate limit 30 for safety)
!python scripts/run_arq_batch.py --input data/unlabeled.csv --rate 30

# 5. Download Results
# The output file 'data/batch_arq_results.csv' will appear in the Output tab
```

### Human Review (Tier 4)
Manage the review queue for ambiguous cases:

```bash
# Show statistics
./venv/bin/python scripts/review.py --stats

# Interactive review mode
./venv/bin/python scripts/review.py --input data/batch_arq_full_details.json
```

## Dependencies

- `openai`: LLM Inference (NVIDIA NIM compatible)
- `sentence-transformers`: Embeddings (all-MiniLM-L6-v2)
- `faiss-cpu`: Vector Search
- `loguru`: Logging
- `numpy`, `pandas`: Data Processing

## References

- [MAFA: A Multi-Agent Framework for Enterprise-Scale Annotation](https://arxiv.org/abs/2510.14184)
- [UIT-ViOCD Dataset](https://arxiv.org/abs/2104.11969)

## License

MIT
