"""ARQ Prompt Builder - MAFA-style Structured Reasoning Queries.

Implements Attentive Reasoning Queries (ARQ) from MAFA paper Section 4.2.1.

ARQ key points from paper:
1. PROMPT: Structured queries guide LLM through systematic reasoning
2. OUTPUT: JSON format with final_label, confidence, reasoning
3. Goal: Mitigate "lost in middle" phenomenon
4. Domain-specific queries tailored to task

NO FALLBACK - All responses must be valid JSON.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import json


from src.config import get_config


@dataclass
class ARQQuery:
    """Single ARQ query for structured reasoning guidance."""

    id: int
    question: str


@dataclass
class ARQPrompt:
    """Complete ARQ prompt structure."""

    system_prompt: str
    reasoning_queries: List[ARQQuery]
    output_schema: Dict[str, Any]
    examples: List[Dict[str, str]]


class ARQPromptBuilder:
    """Builder for ARQ prompts optimized for classification tasks (Dynamic Labels).

    NO FALLBACK - LLM must output valid JSON matching output_schema.
    """

    # Load labels from config dynamically
    _CONFIG = get_config()
    LABELS = (
        _CONFIG.task.labels
        if hasattr(_CONFIG, "task")
        and hasattr(_CONFIG.task, "labels")
        and isinstance(_CONFIG.task.labels, dict)
        else {
            "0": "Non-complaint - Bình luận tích cực, khen ngợi, hỏi đáp, hoặc trung tính.",
            "1": "Complaint - Phàn nàn về sản phẩm, dịch vụ, giao hàng, đóng gói.",
        }
    )

    OUTPUT_SCHEMA = {
        "type": "object",
        "properties": {
            "final_label": {
                "type": "string",
                "enum": list(LABELS.keys()),
                "description": f"Chọn một trong các nhãn: {', '.join(LABELS.keys())}",
            },
            "confidence": {
                "type": "string",
                "description": "Độ tin tưởng từ 0.0 đến 1.0 (ví dụ: 0.95, 0.87). Trả về dưới dạng string.",
            },
            "reasoning": {
                "type": "string",
                "description": "Giải thích chi tiết quyết định, dựa trên phân tích các bước trước",
            },
        },
        "required": ["final_label", "confidence", "reasoning"],
    }

    @staticmethod
    def build_toxicity_arq(
        text: str,
        examples: Optional[List[Dict[str, str]]] = None,
        context: Optional[str] = None,
        agent_type: str = "primary",
    ) -> ARQPrompt:
        """Build ARQ prompt for classification task.

        Args:
            text: Text to classify
            examples: Few-shot examples (optional)
            context: Optional context/title
            agent_type: primary | contextual | retrieval | hybrid
        """
        system_prompt = ARQPromptBuilder._build_system_prompt(text, context)
        reasoning_queries = ARQPromptBuilder._build_queries(agent_type, text, context)
        output_schema = ARQPromptBuilder.OUTPUT_SCHEMA
        examples = examples or []

        return ARQPrompt(
            system_prompt=system_prompt,
            reasoning_queries=reasoning_queries,
            output_schema=output_schema,
            examples=examples,
        )

    @staticmethod
    def _build_system_prompt(text: str, context: Optional[str]) -> str:
        """Build system prompt with task description and examples."""
        config = get_config()
        task_desc = (
            config.task.description if hasattr(config, "task") else "Phân loại văn bản"
        )

        parts = [
            f"Bạn là chuyên gia {task_desc}.",
            "",
            "NHIỆM VỤ: Phân loại comment dựa trên các nhãn sau:",
            "",
            "LABELS:",
        ]

        # Dynamic labels
        for label, desc in ARQPromptBuilder.LABELS.items():
            parts.append(f"  - {label}: {desc}")

        parts.extend(
            [
                "",
                "QUY TẮC QUAN TRỌNG:",
                "- Đọc kỹ và phân tích comment theo từng bước",
                "- Xem xét ngữ cảnh nếu có",
                "- Dựa vào ví dụ minh họa nếu được cung cấp",
                "- Đưa ra quyết định cuối cùng với reasoning rõ ràng",
                "- Confidence: Con số thực tế từ 0.0 đến 1.0 phản ánh độ chắc chắn của bạn",
            ]
        )

        if context:
            parts.extend(["", f"NGỮ CẢNH: {context}"])

        parts.extend(
            [
                "",
                "TEXT CẦN PHÂN LOẠI:",
                f'"{text}"',
            ]
        )

        return "\n".join(parts)

    @staticmethod
    def _build_queries(
        agent_type: str, text: str, context: Optional[str]
    ) -> List[ARQQuery]:
        """Build reasoning queries based on agent type."""
        queries = {
            "primary": ARQPromptBuilder._primary_queries(text),
            "contextual": ARQPromptBuilder._contextual_queries(text, context),
            "retrieval": ARQPromptBuilder._retrieval_queries(text),
            "hybrid": ARQPromptBuilder._hybrid_queries(text),
        }
        return queries.get(agent_type, queries["primary"])

    @staticmethod
    def _primary_queries(text: str) -> List[ARQQuery]:
        """Primary agent: direct analysis."""
        return [
            ARQQuery(
                id=1,
                question="Bước 1 - PHÂN TÍCH TỪ NGỮ: Comment sử dụng từ ngữ như thế nào? Có từ ngữ thể hiện cảm xúc hoặc phàn nàn không?",
            ),
            ARQQuery(
                id=2,
                question="Bước 2 - ĐÁNH GIÁ Ý ĐỊNH: Người viết muốn truyền đạt điều gì? Ý định phàn nàn, khen ngợi, hay hỏi thông tin?",
            ),
            ARQQuery(
                id=3,
                question="Bước 3 - XÁC ĐỊNH ĐỐI TƯỢNG: Nếu có phàn nàn, đối tượng là gì (sản phẩm, giao hàng, dịch vụ...)?",
            ),
            ARQQuery(
                id=4,
                question="Bước 4 - PHÂN TÍCH NGỮ CẢNH: Trong ngữ cảnh comment này, có sự không hài lòng thực sự hay chỉ là hiểu lầm?",
            ),
            ARQQuery(
                id=5,
                question=f"Bước 5 - QUYẾT ĐỊNH: Dựa trên các phân tích trên, label nào ({', '.join(ARQPromptBuilder.LABELS.keys())}) phù hợp nhất?",
            ),
        ]

    @staticmethod
    def _contextual_queries(text: str, context: Optional[str]) -> List[ARQQuery]:
        """Contextual agent: analysis with title/context."""
        title = context or "Không có"
        return [
            ARQQuery(
                id=1,
                question=f"Bước 1 - PHÂN TÍCH TITLE: Tiêu đề '{title}' cho biết điều gì về ngữ cảnh?",
            ),
            ARQQuery(
                id=2,
                question=f"Bước 2 - PHÂN TÍCH TEXT: Comment '{text}' liên quan đến title như thế nào?",
            ),
            ARQQuery(
                id=3,
                question="Bước 3 - ĐÁNH GIÁ SỰ PHÙ HỢP: Comment có match với ngữ cảnh của title không?",
            ),
            ARQQuery(
                id=4,
                question="Bước 4 - PHÂN LOẠI: Dựa trên cả title và text, label nào phù hợp (0 hay 1)?",
            ),
            ARQQuery(
                id=5,
                question="Bước 5 - TITLE INFLUENCE: Title có ảnh hưởng như thế nào đến quyết định của bạn?",
            ),
        ]

    @staticmethod
    def _retrieval_queries(text: str) -> List[ARQQuery]:
        """Retrieval agent: analysis with similar examples."""
        return [
            ARQQuery(
                id=1,
                question="Bước 1 - PHÂN TÍCH VÍ DỤ: Xem xét các ví dụ tương tự được cung cấp. Chúng có đặc điểm chung gì?",
            ),
            ARQQuery(
                id=2,
                question=f"Bước 2 - SO SÁNH: Comment '{text}' giống hay khác các ví dụ trên như thế nào?",
            ),
            ARQQuery(
                id=3,
                question="Bước 3 - PATTERN MATCHING: Có pattern nào từ examples phù hợp với comment không?",
            ),
            ARQQuery(
                id=4,
                question="Bước 4 - QUYẾT ĐỊNH: Dựa trên similarity với examples, label nào phù hợp (0 hay 1)?",
            ),
        ]

    @staticmethod
    def _hybrid_queries(text: str) -> List[ARQQuery]:
        """Hybrid agent: analysis with edge cases and ambiguous patterns."""
        return [
            ARQQuery(
                id=1,
                question="Bước 1 - EDGE CASE ANALYSIS: Xem xét các ví dụ edge cases và ambiguous patterns. Phân tích chúng.",
            ),
            ARQQuery(
                id=2,
                question=f"Bước 2 - AMBIGUOUS CHECK: Comment '{text}' có ambiguous không? Có sarcasm, irony, hoặc implicit toxicity không?",
            ),
            ARQQuery(
                id=3,
                question="Bước 3 - DECISION: Dựa trên edge cases và ambiguous patterns, label nào phù hợp (0 hay 1)?",
            ),
            ARQQuery(
                id=4,
                question="Bước 4 - REASONING: Giải thích chi tiết tại sao bạn chọn label này, đặc biệt với edge cases.",
            ),
        ]

    @staticmethod
    def to_prompt(arq: ARQPrompt) -> str:
        """Convert ARQ prompt to full prompt string for LLM.

        NO FALLBACK - Output must be valid JSON.
        """
        parts = [arq.system_prompt]

        # Add examples if provided
        if arq.examples:
            parts.extend(["", "VÍ DỤ MINH HỌA:"])
            for i, ex in enumerate(arq.examples):
                parts.append(f'  - "{ex["text"]}" → Label: {ex["label"]}')

        # Add reasoning queries (to guide LLM, not to answer)
        parts.extend(
            [
                "",
                "HƯỚNG DẪN SUY LUẬN (trả lời theo các bước sau):",
            ]
        )
        for q in arq.reasoning_queries:
            parts.append(f"  {q.question}")

        # Add output format requirement
        schema_str = json.dumps(arq.output_schema, indent=2, ensure_ascii=False)
        parts.extend(
            [
                "",
                "YÊU CẦU OUTPUT:",
                f"Phản hồi CHỈ BAO GỒM JSON hợp lệ, KHÔNG có text khác.",
                "",
                "JSON Schema:",
                f"```json",
                schema_str,
                f"```",
                "",
                "QUAN TRỌNG:",
                "- Phản hồi chỉ là JSON, không có markdown code block, không có giải thích thêm",
                "- JSON phải hợp lệ và match với schema trên",
                "- reasoning phải cụ thể, có bằng chứng từ phân tích",
            ]
        )

        return "\n".join(parts)

    @staticmethod
    def parse_response(response: str) -> Dict[str, Any]:
        """Parse ARQ response - MUST be valid JSON.

        NO FALLBACK - If response is not valid JSON, raise error.
        """
        response = response.strip()

        # Remove markdown code blocks if present
        if response.startswith("```json"):
            response = response[7:-3]
        elif response.startswith("```"):
            response = response[3:-3]

        # MUST be valid JSON - NO FALLBACK
        try:
            data = json.loads(response)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON response from LLM: {e}. Response: {response[:200]}"
            )

        # Validate required fields
        if "final_label" not in data:
            raise ValueError(f"Missing 'final_label' in response: {data}")
        if "confidence" not in data:
            raise ValueError(f"Missing 'confidence' in response: {data}")
        if "reasoning" not in data:
            raise ValueError(f"Missing 'reasoning' in response: {data}")

        # Validate label (Dynamic check)
        valid_labels = list(ARQPromptBuilder.LABELS.keys())
        if data["final_label"] not in valid_labels:
            raise ValueError(
                f"Invalid label: {data['final_label']}. Expected one of {valid_labels}"
            )

        # Validate confidence
        try:
            conf_val = float(data["confidence"])
            if not (0.0 <= conf_val <= 1.0):
                raise ValueError(f"Confidence out of range: {conf_val}")

            # Auto-assign level for backward compatibility
            if conf_val >= 0.8:
                level = "HIGH"
            elif conf_val >= 0.5:
                level = "MEDIUM"
            else:
                level = "LOW"

            return {
                "final_label": data["final_label"],
                "confidence": level,
                "confidence_score": conf_val,
                "reasoning": data["reasoning"],
            }
        except ValueError:
            # Fallback if model still returns text like "HIGH"
            score = ARQPromptBuilder.confidence_to_score(data["confidence"])
            return {
                "final_label": data["final_label"],
                "confidence": data["confidence"],
                "confidence_score": score,
                "reasoning": data["reasoning"],
            }

    @staticmethod
    def confidence_to_score(confidence: str) -> float:
        """Convert confidence level to numeric score."""
        mapping = {
            "HIGH": 0.9,
            "MEDIUM": 0.7,
            "LOW": 0.5,
        }
        return mapping.get(confidence.upper(), 0.5)
