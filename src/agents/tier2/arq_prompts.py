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

    @staticmethod
    def _get_labels() -> Dict[str, str]:
        """Get labels dynamically from config."""
        config = get_config()
        if (
            hasattr(config, "task")
            and hasattr(config.task, "labels")
            and isinstance(config.task.labels, dict)
        ):
            return config.task.labels

        # Default fallback if config missing (should not happen in prod)
        return {
            "0": "Non-complaint - Khen ngợi thuần túy, hài lòng, tích cực.",
            "1": "Complaint - Phàn nàn, góp ý, không hài lòng về sản phẩm/dịch vụ/giao hàng.",
        }

    @staticmethod
    def _get_output_schema() -> Dict[str, Any]:
        """Get output schema dynamically based on current labels."""
        labels = ARQPromptBuilder._get_labels()
        return {
            "type": "object",
            "properties": {
                "final_label": {
                    "type": "string",
                    "enum": list(labels.keys()),
                    "description": f"Chọn một trong các nhãn: {', '.join(labels.keys())}",
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
        """Build ARQ prompt for classification task."""
        system_prompt = ARQPromptBuilder._build_system_prompt(text, context)
        reasoning_queries = ARQPromptBuilder._build_queries(agent_type, text, context)
        output_schema = ARQPromptBuilder._get_output_schema()
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
        labels = ARQPromptBuilder._get_labels()
        for label, desc in labels.items():
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
        labels = ARQPromptBuilder._get_labels()
        return [
            ARQQuery(
                id=1,
                question="Bước 1 - PHÂN TÍCH CẢM XÚC: Comment thể hiện sự hài lòng (khen) hay không hài lòng (chê/phàn nàn/góp ý)?",
            ),
            ARQQuery(
                id=2,
                question="Bước 2 - TÌM DẤU HIỆU PHÀN NÀN: Có từ/cụm từ thể hiện sự thất vọng, mong muốn cải thiện, góp ý, cảnh báo không? (VD: 'nhưng', 'tuy nhiên', 'phải chi', 'giá mà', 'hơi', 'chưa', 'không được', 'lâu', 'chậm', 'thiếu'...)",
            ),
            ARQQuery(
                id=3,
                question="Bước 3 - KIỂM TRA MIXED CONTENT: Comment có VỪA khen VỪA chê không? (VD: 'đẹp nhưng hơi nhỏ', 'tốt nhưng giao chậm'). Nếu có BẤT KỲ phàn nàn/góp ý nào → Label 1.",
            ),
            ARQQuery(
                id=4,
                question="Bước 4 - QUYẾT ĐỊNH: Label 1 = Có phàn nàn/góp ý/chê (kể cả mixed). Label 0 = Chỉ khen ngợi thuần túy, không có bất kỳ phàn nàn nào.",
            ),
        ]

    @staticmethod
    def _contextual_queries(text: str, context: Optional[str]) -> List[ARQQuery]:
        """Critic agent (formerly Contextual): Devil's Advocate analysis based on Olshtain & Weinbach.

        Focuses on verifying strictly against the definition constraints:
        - Complaint (1) shows dissatisfaction, unmet expectation, suggestion, warning.
        - Non-complaint (0) is PURE satisfaction/praise with NO complaints.
        """
        return [
            ARQQuery(
                id=1,
                question="Bước 1 - TÌM PHÀN NÀN ẨN: Đọc kỹ từng từ. Có bất kỳ dấu hiệu không hài lòng nào không? (giao chậm, thiếu, lỗi, không đúng, hơi, chưa...)",
            ),
            ARQQuery(
                id=2,
                question="Bước 2 - CHECK WISH/SUGGESTION: Tìm các cấu trúc 'Giá mà', 'Phải chi', 'Mong', 'Nên', 'Cần'... Đây là dấu hiệu của Complaint (Label 1).",
            ),
            ARQQuery(
                id=3,
                question="Bước 3 - CHECK MIXED CONTENT: Có pattern 'khen + nhưng/tuy nhiên + chê' không? VD: 'đẹp nhưng mỏng', 'tốt nhưng giao lâu'. Mixed = Label 1.",
            ),
            ARQQuery(
                id=4,
                question="Bước 4 - PHÁN QUYẾT: Nếu có BẤT KỲ phàn nàn/góp ý/wish nào → Label 1. Chỉ khi THUẦN khen ngợi, không có gì tiêu cực → Label 0.",
            ),
        ]

    @staticmethod
    def _retrieval_queries(text: str) -> List[ARQQuery]:
        """Retrieval agent: analysis with similar examples."""
        return [
            ARQQuery(
                id=1,
                question="Bước 1 - PHÂN TÍCH VÍ DỤ: Các ví dụ tương tự được gán nhãn như thế nào? Chú ý pattern: có phàn nàn → Label 1, chỉ khen → Label 0.",
            ),
            ARQQuery(
                id=2,
                question=f"Bước 2 - SO SÁNH: Comment '{text}' giống ví dụ nào hơn? Ví dụ có phàn nàn/góp ý (Label 1) hay ví dụ thuần khen (Label 0)?",
            ),
            ARQQuery(
                id=3,
                question="Bước 3 - PATTERN MATCHING: Comment có chứa từ khóa phàn nàn (chậm, lâu, thiếu, lỗi, không đúng, hơi...) không?",
            ),
            ARQQuery(
                id=4,
                question="Bước 4 - QUYẾT ĐỊNH: Dựa trên similarity với ví dụ. Label 1 = Có phàn nàn. Label 0 = Thuần khen ngợi.",
            ),
        ]

    @staticmethod
    def _hybrid_queries(text: str) -> List[ARQQuery]:
        """Hybrid agent: analysis with edge cases and ambiguous patterns."""
        return [
            ARQQuery(
                id=1,
                question="Bước 1 - EDGE CASE: Kiểm tra các trường hợp khó: 'Khen nhưng có góp ý nhẹ' (Label 1) vs 'Khen thuần túy' (Label 0).",
            ),
            ARQQuery(
                id=2,
                question=f"Bước 2 - AMBIGUOUS CHECK: Comment '{text}' có ẩn ý phàn nàn không? (VD: 'được', 'tạm', 'cũng ok' có thể là khen miễn cưỡng)",
            ),
            ARQQuery(
                id=3,
                question="Bước 3 - DECISION: Nếu có BẤT KỲ dấu hiệu không hài lòng, góp ý, wish → Label 1. Chỉ thuần khen → Label 0.",
            ),
            ARQQuery(
                id=4,
                question="Bước 4 - REASONING: Giải thích rõ tại sao chọn label, đặc biệt với các comment mixed hoặc ambiguous.",
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
        valid_labels = list(ARQPromptBuilder._get_labels().keys())
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
