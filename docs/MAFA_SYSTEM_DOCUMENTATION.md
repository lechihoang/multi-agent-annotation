# MAFA-ViOCD: Multi-Agent Framework for Vietnamese Complaint Detection

## Tài liệu kỹ thuật chi tiết

---

## 1. Tổng quan hệ thống

### 1.1. Kiến trúc 4-Tier

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

### 1.2. Luồng xử lý

1. **Tier 1**: Phân tích và mở rộng query
2. **Tier 2**: 4 agents chạy song song, mỗi agent có chiến lược khác nhau
3. **Tier 3**: Judge tổng hợp kết quả bằng weighted voting
4. **Tier 4**: Human review cho các trường hợp confidence thấp

---

## 2. Tier 1: Query Planning

### 2.1. Query Expander

**Mục đích**: Mở rộng query ngắn/ mơ hồ để cải thiện context cho downstream tasks.

**Cách hoạt động**:
- Sử dụng embedding-based similarity search
- Tìm các câu tương tự trong training data
- Mở rộng query với thông tin từ các câu gần nhất

**Tham số**:
```python
top_k = 5  # Số câu tương tự để mở rộng
```

### 2.2. Query Planner

**Mục đích**: Phân tích intent và lập kế hoạch truy vấn.

**Output**:
- Task type (classification/ner/sentiment)
- Routing decision
- Label names

---

## 3. Tier 2: Specialized Agents

Tất cả agents đều sử dụng **ARQ (Attentive Reasoning Queries)** - hướng dẫn LLM qua các bước suy luận có hệ thống.

### 3.1. Primary Agent (Agent A)

**File**: `src/agents/tier2/primary_only.py`

**Chiến lược**: Phân loại trực tiếp dựa trên định nghĩa nghiêm ngặt của Olshtain & Weinbach.

**Prompt Structure**:

```python
# System Prompt
"""Bạn là chuyên gia {task_description}.

NHIỆM VỤ: Phân loại comment dựa trên các nhãn sau:

LABELS:
  - 0: {label_0_description}
  - 1: {label_1_description}

QUY TẮC QUAN TRỌNG:
- Đọc kỹ và phân tích comment theo từng bước
- Xem xét ngữ cảnh nếu có
- Dựa vào ví dụ minh họa nếu được cung cấp
- Đưa ra quyết định cuối cùng với reasoning rõ ràng
- Confidence: Con số thực tế từ 0.0 đến 1.0

TEXT CẦN PHÂN LOẠI:
"{text}"
"""

# ARQ Reasoning Queries
Bước 1 - PHÂN TÍCH NỘI DUNG: 
Comment thể hiện (a) hài lòng/khen, (b) không hài lòng/phàn nàn/góp ý, 
hay (c) chửi bới/xúc phạm/hate speech?

Bước 2 - KIỂM TRA TÍNH XÂY DỰNG (Constructive): 
Nếu comment tiêu cực, nó có MANG TÍNH XÂY DỰNG không? 
Tức là: thể hiện kỳ vọng chưa được đáp ứng, góp ý cải thiện, cảnh báo, mong muốn (wish), gợi ý? 
Hay chỉ là chửi bới/xúc phạm/hate speech KHÔNG giúp doanh nghiệp cải thiện?
(VD: 'Game như L** đừng tải' = xúc phạm, KHÔNG xây dựng → Label 0. 
     'Giao hàng chậm quá' = góp ý xây dựng → Label 1)

Bước 3 - KIỂM TRA MIXED CONTENT: 
Comment có VỪA khen VỪA chê không? 
(VD: 'đẹp nhưng hơi nhỏ', 'tốt nhưng giao chậm'). 
Nếu có phàn nàn/góp ý mang tính xây dựng → Label 1.

Bước 4 - QUYẾT ĐỊNH: 
Label 1 = Phàn nàn/góp ý/wish/cảnh báo MANG TÍNH XÂY DỰNG (kể cả mixed khen+chê). 
Label 0 = Khen ngợi thuần túy, hài lòng HOẶC chửi bới/xúc phạm/hate speech KHÔNG mang tính xây dựng.

# Output Format
YÊU CẦU OUTPUT:
Phản hồi CHỈ BAO GỒM JSON hợp lệ, KHÔNG có text khác.

JSON Schema:
{
  "final_label": "0 hoặc 1",
  "confidence": "0.0 đến 1.0 (string)",
  "reasoning": "Giải thích chi tiết quyết định..."
}
```

### 3.2. Critic Agent (Agent B) - Devil's Advocate

**File**: `src/agents/tier2/contextual.py`

**Chiến lược**: Kiểm tra nghiêm ngặt để tránh false positives. Đóng vai "Devil's Advocate".

**Prompt Structure**:

```python
# System Prompt (tương tự Primary + context)
"""Bạn là chuyên gia {task_description}.

NHIỆM VỤ: Phân tích comment một cách NGHIÊM NGẶT để tìm phàn nàn ẩn.

LABELS:
  - 0: {label_0_description}
  - 1: {label_1_description}

QUY TẮC QUAN TRỌNG:
- Đọc KỸ từng từ, tìm dấu hiệu không hài lòng ẩn
- Phân biệt: phàn nàn xây dựng vs chửi bới không xây dựng
- Wish/suggestion là dấu hiệu của complaint

TEXT CẦN PHÂN TÍCH:
"{text}"
"""

# ARQ Reasoning Queries (Devil's Advocate style)
Bước 1 - TÌM PHÀN NÀN ẨN: 
Đọc kỹ từng từ. Có bất kỳ dấu hiệu không hài lòng nào không? 
(giao chậm, thiếu, lỗi, không đúng, hơi, chưa...)

Bước 2 - CHECK WISH/SUGGESTION: 
Tìm các cấu trúc 'Giá mà', 'Phải chi', 'Mong', 'Nên', 'Cần'... 
Đây là dấu hiệu của Complaint (Label 1).

Bước 3 - KIỂM TRA TÍNH XÂY DỰNG: 
Nếu comment tiêu cực, phân biệt: 
(a) Phàn nàn/góp ý mang tính xây dựng (kỳ vọng chưa đáp ứng, mong cải thiện) → Label 1
(b) Chửi bới/xúc phạm/hate speech KHÔNG xây dựng (VD: 'Game như L** đừng tải', 'ngu', 'rác') → Label 0
(c) Mixed khen+chê xây dựng → Label 1

Bước 4 - PHÁN QUYẾT: 
Label 1 = Phàn nàn/góp ý/wish MANG TÍNH XÂY DỰNG (kể cả mixed). 
Label 0 = Khen ngợi thuần túy HOẶC chửi bới/xúc phạm/hate speech KHÔNG mang tính xây dựng.
```

### 3.3. Retrieval Agent (Agent C)

**File**: `src/agents/tier2/retrieval.py`

**Chiến lược**: Sử dụng RAG (Retrieval-Augmented Generation) với few-shot examples từ seed dataset.

**Cách hoạt động**:
1. Encode input text và seed examples bằng `sentence-transformers/all-MiniLM-L6-v2`
2. Dùng FAISS để tìm k examples tương tự nhất
3. Đưa examples vào prompt
4. LLM phân loại dựa trên pattern từ examples

**Tham số**:
```yaml
k_examples: 3  # Số examples để retrieve
embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
faiss_metric: "cosine"
```

**Prompt Structure**:

```python
# System Prompt
"""Bạn là chuyên gia {task_description}.

NHIỆM VỤ: Phân loại comment dựa trên CÁC VÍ DỤ TƯƠNG TỰ.

LABELS:
  - 0: {label_0_description}
  - 1: {label_1_description}

VÍ DỤ TƯƠNG TỰ (từ seed dataset):
{examples}

TEXT CẦN PHÂN LOẠI:
"{text}"
"""

# ARQ Reasoning Queries
Bước 1 - PHÂN TÍCH VÍ DỤ: 
Các ví dụ tương tự được gán nhãn như thế nào? 
Chú ý pattern: phàn nàn xây dựng → Label 1, khen thuần/chửi bới không xây dựng → Label 0.

Bước 2 - SO SÁNH: 
Comment '{text}' giống ví dụ nào hơn? 
Ví dụ có phàn nàn/góp ý xây dựng (Label 1), ví dụ khen (Label 0), hay ví dụ chửi bới/xúc phạm không xây dựng (Label 0)?

Bước 3 - KIỂM TRA TÍNH XÂY DỰNG: 
Comment có thể hiện kỳ vọng chưa đáp ứng, mong muốn cải thiện, góp ý cụ thể không? (→ Label 1). 
Hay chỉ là chửi bới/xúc phạm chung chung không giúp cải thiện? (→ Label 0).

Bước 4 - QUYẾT ĐỊNH: 
Dựa trên similarity với ví dụ. 
Label 1 = Phàn nàn/góp ý MANG TÍNH XÂY DỰNG. 
Label 0 = Khen ngợi thuần túy HOẶC chửi bới/hate speech KHÔNG xây dựng.
```

### 3.4. Hybrid Agent (Agent D) - Retrieval MRL

**File**: `src/agents/tier2/retrieval_mrl.py`

**Chiến lược**: Xử lý edge cases và ambiguous patterns với **DIFFERENT examples** (khác Agent C).

**Đặc điểm**:
- Dùng cùng embedding model với Agent C
- Nhưng với **UNIQUE examples** tập trung vào edge cases
- Examples: ambiguous patterns, sarcasm, teencode, cross-topic

**Focus**:
- Sarcasm detection
- Teencode handling  
- Implicit negation
- Mixed sentiment
- Ambiguous language

**Prompt Structure**:

```python
# System Prompt
"""Bạn là chuyên gia {task_description}, chuyên xử lý CÁC TRƯỜNG HỢP KHÓ.

NHIỆM VỤ: Phân loại comment, ĐẶC BIỆT CHÚ Ý các trường hợp edge cases.

LABELS:
  - 0: {label_0_description}
  - 1: {label_1_description}

VÍ DỤ EDGE CASES:
{unique_examples}

TEXT CẦN PHÂN LOẠI:
"{text}"
"""

# ARQ Reasoning Queries (Edge-case focused)
Bước 1 - EDGE CASE: 
Kiểm tra các trường hợp khó: 
(a) 'Khen nhưng có góp ý nhẹ' → Label 1
(b) 'Khen thuần túy' → Label 0
(c) 'Chửi bới/xúc phạm không xây dựng' (VD: 'Game như L** đừng tải', 'ngu', 'rác') → Label 0

Bước 2 - AMBIGUOUS CHECK: 
Comment có ẩn ý phàn nàn không? 
(VD: 'được', 'tạm', 'cũng ok' có thể là khen miễn cưỡng). 
Phân biệt: phàn nàn xây dựng (kỳ vọng chưa đáp ứng) vs chửi bới/hate speech (không xây dựng).

Bước 3 - DECISION: 
Label 1 = Phàn nàn/góp ý/wish MANG TÍNH XÂY DỰNG (kể cả mixed). 
Label 0 = Khen thuần túy HOẶC chửi bới/xúc phạm/hate speech KHÔNG xây dựng.

Bước 4 - REASONING: 
Giải thích rõ tại sao chọn label. 
Nếu comment tiêu cực, giải thích nó xây dựng (→ Label 1) hay chỉ xúc phạm (→ Label 0).
```

---

## 4. Tier 1: Query Planning (Bổ sung)

### 4.1. Query Expander

**File**: `src/agents/tier1/query_expander.py`

**Chiến lược**: Embedding-based query expansion (không dùng LLM để tăng tốc độ).

**Cách hoạt động**:
1. Load vocabulary từ training data
2. Encode vocabulary bằng `sentence-transformers/all-MiniLM-L6-v2`
3. Tìm các terms tương tự nhất với input
4. Mở rộng query với các terms liên quan

**Tham số**:
```python
max_vocab_size = 10000
top_k = 5  # Số terms để mở rộng
```

**Ví dụ**:
```
Input:  "tôi thấy ngườ lái xe hơi bấm còi"
Output: "tôi thấy ngườ lái xe hơi bấm còi lái xe giao thông vi phạm"
```

### 4.2. Query Planner

**File**: `src/agents/tier1/query_planner.py`

**Chiến lược**: LLM-based intent analysis và contextual expansion.

**Prompt**:
```python
"""Bạn là chuyên gia phân tích và mở rộng query cho hệ thống phân loại văn bản.

QUERY: {query}

NHIỆM VỤ:
1. Phân tích intent chính của comment
2. Mở rộng query với các từ khóa liên quan (đồng nghĩa, ngữ cảnh)
3. Trả về expanded query để giúp search engine hoặc model hiểu rõ ngữ nghĩa

QUY TẮC:
- Giữ nguyên ý nghĩa gốc
- Thêm các từ khóa làm rõ ngữ cảnh (ví dụ: nếu khen sản phẩm -> thêm "tích cực", "hài lòng")
- Không thay đổi quan điểm của ngườ viết

Respond JSON:
{"intent": "...", "expanded_query": "..."}"""
```

### 4.3. Router Agent

**File**: `src/agents/tier1/router.py`

**Chiến lược**: Dynamic task routing dựa trên TaskParser.

**Prompt (fallback)**:
```python
"""You are a classification expert. Classify the text into ONE of the provided labels.

LABELS: {labels_str}

TEXT: {text}

Respond ONLY with valid JSON:
{"label": "...", "confidence": 0.0-1.0, "reasoning": "..."}"""
```

---

## 4. Tier 3: Judge Agent & Consensus

### 4.1. Weighted Voting Algorithm

**Công thức** (MAFA Section 4.5):

```
weight[i] = accuracy[i] / Σ(accuracy[j])

Final_Score(candidate) = Σ(weight[i] × confidence[i] × score[i][candidate])
```

### 4.2. Confidence Calibration

**Calibration Factors**:
```python
CONFIDENCE_FACTORS = {
    "HIGH": 1.5,    # confidence >= 0.8
    "MEDIUM": 1.0,  # 0.5 <= confidence < 0.8
    "LOW": 0.5      # confidence < 0.5
}
```

**Calibrated Score**:
```
calibrated_score = confidence × conf_factor × hist_weight
```

### 4.3. Agreement Bonus

```
agreement_score = (số agents đồng ý label phổ biến nhất) / (tổng số agents)
agreement_bonus = agreement_score × 0.1  # 10% bonus
```

### 4.4. Final Score

```
raw_score = Σ(calibrated_scores) / n_agents
final_score = min(raw_score + agreement_bonus, 1.0)
```

### 4.5. Decision Thresholds

```yaml
thresholds:
  approve: 0.85    # Auto-approve
  review: 0.60     # Human review
  escalate: <0.60  # Expert escalation
```

**Decision Logic**:
```python
if final_score >= 0.85:
    decision = "approve"
elif final_score >= 0.60:
    decision = "review"
else:
    decision = "escalate"
```

### 4.6. Dynamic Weights

**Default Weights**:
```python
DEFAULT_WEIGHTS = {
    "primary_only": 0.25,
    "contextual": 0.25,
    "retrieval": 0.25,
    "retrieval_mrl": 0.25
}
```

**Dynamic Update** (từ MetricsCollector):
```python
weight[i] = accuracy[i] / sum(accuracy)
```

---

## 5. Tier 4: Human Review

### 5.1. Review Queue

**Điều kiện vào queue**:
- Confidence < 0.85
- Agents không đồng thuận (agreement < 0.75)

### 5.2. Feedback Loop

Human corrections được dùng để:
1. Cập nhật agent weights
2. Cải thiện few-shot examples
3. Tinh chỉnh prompts

---

## 6. Cấu hình hệ thống

### 6.1. config.yaml

```yaml
# Provider configuration
provider:
  type: "nvidia"

# NVIDIA NIM API settings
nvidia:
  enabled: true
  model: "meta/llama-3.3-70b-instruct"
  temperature: 0.1
  max_tokens: 1024

# Embedding
embedding:
  model: "sentence-transformers/all-MiniLM-L6-v2"
  use_local: true

# FAISS
faiss:
  dimension: 384
  metric: "cosine"

# Agent weights
agents:
  weights:
    primary_only: 0.25
    contextual: 0.25
    retrieval: 0.25
    hybrid: 0.25
  retrieval:
    k_examples: 3

# Thresholds
thresholds:
  approve: 0.85
  review: 0.60

# Task Configuration
task:
  type: "classification"
  name: "complaint_detection"
  description: "Phân loại phàn nàn trong bình luận thương mại điện tử tiếng Việt."
  
  labels:
    "0": "Non-complaint - Khen ngợi, hài lòng, tích cực..."
    "1": "Complaint - Phàn nàn, góp ý, mong muốn..."
```

### 6.2. Môi trường

```bash
# Required environment variables
NVIDIA_API_KEY=nvapi-...
# hoặc
GROQ_API_KEY=gsk_...
```

---

## 7. Định nghĩa nhãn

### 7.1. Label 0: Non-complaint

**Đặc điểm**:
- Khen ngợi thuần túy
- Hài lòng về sản phẩm/dịch vụ
- Tích cực, không có phàn nàn

**Bao gồm**:
- Chửi bới/Hateful speech KHÔNG mang tính xây dựng
  - VD: "Game như L** đừng tải", "ngu", "rác"

### 7.2. Label 1: Complaint

**Đặc điểm**:
- Phàn nàn, góp ý
- Mong muốn (Wish)
- Cảnh báo
- Thể hiện sự không hài lòng giữa kỳ vọng và thực tế
- MANG TÍNH XÂY DỰNG

**Bao gồm**:
- Mixed content (vừa khen vừa chê)
  - VD: "đẹp nhưng hơi nhỏ", "tốt nhưng giao chậm"

**Dấu hiệu nhận biết**:
- Cấu trúc wish: "Giá mà", "Phải chi", "Mong", "Nên", "Cần"
- Từ khóa: giao chậm, thiếu, lỗi, không đúng, hơi, chưa

---

## 8. Prompt Engineering

### 8.1. Nguyên tắc ARQ (Attentive Reasoning Queries)

1. **Structured queries**: Hướng dẫn LLM qua các bước suy luận có hệ thống
2. **Output JSON**: Bắt buộc format chuẩn, không có fallback
3. **Mitigate "lost in middle"**: Lặp lại key instructions ở mỗi bước
4. **Domain-specific**: Câu hỏi tailored cho từng task

### 8.2. System Prompt Template

```
Bạn là chuyên gia {task_description}.

NHIỆM VỤ: Phân loại comment dựa trên các nhãn sau:

LABELS:
  - 0: {label_0_description}
  - 1: {label_1_description}

QUY TẮC QUAN TRỌNG:
- Đọc kỹ và phân tích comment theo từng bước
- Xem xét ngữ cảnh nếu có
- Dựa vào ví dụ minh họa nếu được cung cấp
- Đưa ra quyết định cuối cùng với reasoning rõ ràng
- Confidence: Con số thực tế từ 0.0 đến 1.0

TEXT CẦN PHÂN LOẠI:
"{text}"
```

### 8.3. Output Format

```json
{
  "final_label": "0 hoặc 1",
  "confidence": "0.95",
  "reasoning": "Giải thích chi tiết quyết định..."
}
```

**Yêu cầu**:
- Chỉ trả về JSON, không có text khác
- Không dùng markdown code block
- JSON phải hợp lệ và match schema

---

## 9. Evaluation Metrics

### 9.1. Agent-level Metrics

- **Accuracy**: % predictions đúng
- **Confidence Calibration**: Mức độ khớp giữa confidence và accuracy
- **Agreement Rate**: % trường hợp đồng ý với consensus

### 9.2. System-level Metrics

- **Consensus Score**: Điểm tổng hợp từ Judge
- **Decision Distribution**: % approve/review/escalate
- **Human Review Rate**: % cần human intervention

### 9.3. Data Quality Metrics

- **Inter-annotator Agreement**: Độ đồng thuận giữa agents
- **Label Distribution**: Cân bằng giữa các classes
- **Confidence Distribution**: Phân bố độ tin cậy

---

## 10. File Structure

```
Multi-agent-annotaton/
├── config.yaml              # Cấu hình hệ thống
├── README.md                # Tài liệu tổng quan
├── requirements.txt         # Dependencies
├── data/
│   ├── seed.csv            # 100 samples đã gán nhãn
│   ├── train.csv           # Train gốc
│   ├── val.csv             # Val gốc
│   ├── test.csv            # Test set
│   ├── unlabeled.csv       # Data cần gán nhãn
│   ├── train_labeled.csv   # Train sau khi annotate
│   ├── val_labeled.csv     # Val sau khi annotate
│   └── batch_arq_results.csv  # Kết quả từ MAFA
├── src/
│   ├── config.py           # Configuration loader
│   ├── main.py             # Pipeline chính
│   ├── agents/
│   │   ├── tier1/          # Query Planning
│   │   │   ├── query_expander.py
│   │   │   └── query_planner.py
│   │   ├── tier2/          # Specialized Agents
│   │   │   ├── arq_prompts.py      # ARQ prompt builder
│   │   │   ├── primary_only.py     # Primary Agent
│   │   │   ├── contextual.py       # Critic Agent
│   │   │   ├── retrieval.py        # Retrieval Agent
│   │   │   └── retrieval_mrl.py    # Hybrid Agent
│   │   ├── tier3/          # Judge
│   │   │   └── judge.py
│   │   └── tier4/          # Human Review
│   │       └── review.py
│   ├── consensus/
│   │   └── voting.py       # Weighted voting
│   └── api/
│       └── nim_client.py   # NVIDIA NIM client
├── scripts/
│   ├── run_arq_batch.py    # Batch runner
│   └── merge_labeled_data.py  # Merge results
└── notebooks/
    ├── viocd_logistic_regression_baseline.ipynb
    └── compare_original_vs_reannotated.ipynb
```

---

## 11. References

1. **MAFA Paper**: arXiv:2510.14184 - Multi-Agent Framework for Data Annotation
2. **Olshtain & Weinbach**: Complaint speech act theory
3. **ARQ**: Attentive Reasoning Queries for mitigating "lost in middle"

---

## 12. Tác giả

Hệ thống MAFA-ViOCD được phát triển cho bài toán Vietnamese Online Complaint Detection.
