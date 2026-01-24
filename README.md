# MAFA-ViOCD: Multi-Agent Framework for Vietnamese Complaint Detection

This repository implements an Enterprise-Scale Annotation System based on the MAFA framework (arXiv:2510.14184), optimized for Vietnamese Online Complaint Detection (ViOCD).

## System Architecture

The system follows a 4-Tier Multi-Agent architecture designed for high accuracy and consistency:

```
User Input Batch
       │
       ▼
[Tier 1: Query Planner] (Query Expansion)
       │
       ├──────────────────────────────────────────────┐
       │                                              │
       ▼                                              ▼
[Tier 2: Specialized Agents (Parallel Execution)]
 ┌─────────────┐  ┌────────────┐  ┌─────────────┐  ┌────────────┐
 │Primary Agent│  │Critic Agent│  │Retrieval Agt│  │Hybrid Agent│
 │(Definition) │  │(Devil's Adv)│ │(Few-shot RAG)│ │(Edge Cases)│
 └──────┬──────┘  └──────┬─────┘  └──────┬──────┘  └──────┬─────┘
        │                │               │                │
        └────────────────┼───────────────┼────────────────┘
                         │
                         ▼
               [Tier 3: Judge Agent] (Consensus & Voting)
                         │
                         ▼
                  Confidence Score?
                 ╱                 ╲
        [High (>=0.85)]       [Low/Med (<0.85)]
               │                       │
               ▼                       ▼
        [Auto Approve]          [Review Queue] ───> [Tier 4: Human Review]
```

### Tier 1: Query Planning
*   **Query Expander:** Analyzes user intent and expands short/ambiguous queries to improve downstream retrieval and classification context.

### Tier 2: Specialized Agents
Four parallel agents analyze the input using distinct strategies:
1.  **Primary Agent:** Performs direct classification based on strict label definitions.
2.  **Critic Agent (formerly Contextual):** Acts as a "Devil's Advocate" to enforce strict adherence to Olshtain & Weinbach's complaint definitions. It specifically checks for:
    *   Hate speech disguised as complaints (should be Label 0).
    *   Constructive criticism hidden in compliments (should be Label 1).
3.  **Retrieval Agent:** Utilizes RAG (Retrieval-Augmented Generation) to find semantically similar examples from the seed dataset, ensuring consistency with historical annotations.
4.  **Hybrid Agent:** Combines retrieval with deep reasoning to handle edge cases like sarcasm, teencode, and implicit negation.

### Tier 3: Judge & Consensus
*   **Judge Agent:** Aggregates outputs from Tier 2 agents using a weighted voting mechanism.
*   **Decision Logic:**
    *   **Approve:** High consensus (Confidence >= 0.85).
    *   **Review:** Moderate consensus or conflicting agents (0.60 <= Confidence < 0.85).
    *   **Escalate:** Low confidence (Confidence < 0.60).

### Tier 4: Human-in-the-Loop
*   **Review Queue:** Captures ambiguous cases for manual verification.
*   **Dynamic Feedback:** Human corrections can be used to update agent weights (planned feature).

## Installation

### Local Setup

1.  Clone the repository:
    ```bash
    git clone <repository_url>
    cd Multi-agent-annotaton
    ```

2.  Create and activate a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate   # Windows
    ```

3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4.  Configure Environment Variables:
    Copy `.env.example` to `.env` and set your API keys (Groq or NVIDIA NIM):
    ```ini
    GROQ_API_KEY=gsk_...
    # or
    NVIDIA_API_KEY=nvapi-...
    ```

## Usage

### Running Batch Annotation

To annotate a large dataset (CSV format):

```bash
python scripts/run_arq_batch.py \
    --input data/unlabeled.csv \
    --output data/results.csv \
    --batch-size 10 \
    --rate 40
```

Arguments:
*   `--input`: Path to input CSV file. Must contain the text column specified in config.
*   `--output`: Path to save results.
*   `--batch-size`: Number of samples per LLM request (Tier 1 optimization).
*   `--max`: Maximum number of samples to process (optional).

### Running on Kaggle / Google Colab

Use the following commands in a notebook cell:

```python
# 1. Setup
!git clone <repository_url>
%cd Multi-agent-annotaton
!pip install -r requirements.txt

# 2. Set API Key
import os
os.environ["GROQ_API_KEY"] = "your_api_key_here"

# 3. Run Annotation
!python scripts/run_arq_batch.py \
    --input data/unlabeled.csv \
    --output results.csv \
    --batch-size 5
```

## Configuration

The system is fully configurable via `config.yaml`.

### Task Definition
Define your classification task and labels:

```yaml
task:
  name: "complaint_detection"
  labels:
    "0": "Non-complaint - Compliments, Neutral, or Hate Speech/Insults"
    "1": "Complaint - Constructive dissatisfaction/Unmet expectations"
  
  columns:
    text: "review"  # Column name in input CSV
```

### Agent Weights
Adjust the voting power of each agent:

```yaml
agents:
  primary: 0.25
  contextual: 0.25  # Applied to Critic Agent
  retrieval: 0.25
  retrieval_mrl: 0.25
```

## Output Format

The output CSV contains the following columns:

*   `text`: Original input text.
*   `final_label`: The consensus label (0 or 1).
*   `confidence`: Confidence score (0.0 - 1.0).
*   `decision`: Action taken (`approve`, `review`, `escalate`).

## References

*   **MAFA Framework:** Hegazy et al., "MAFA: A Multi-Agent Framework for Enterprise-Scale Annotation with Configurable Task Adaptation", arXiv:2510.14184, 2025. [Link](https://arxiv.org/abs/2510.14184)

