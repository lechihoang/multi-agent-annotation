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
            "0": "Non-complaint - Bình luận tích cực, khen ngợi, hỏi đáp, hoặc trung tính.",
            "1": "Complaint - Phàn nàn về sản phẩm, dịch vụ, giao hàng, đóng gói.",
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
                question="Bước 1 - PHÂN TÍCH CẢM XÚC & TỪ NGỮ: Comment thể hiện sự hài lòng (khen) hay không hài lòng (chê)? Có từ ngữ thô tục, chửi bới, xúc phạm (Hateful/Profanity) không?",
            ),
            ARQQuery(
                id=2,
                question="Bước 2 - ĐÁNH GIÁ TÍNH XÂY DỰNG (CONSTRUCTIVE): Nếu là chê, người viết có đưa ra lý do cụ thể, cảnh báo, hay mong muốn giải quyết (Wish) không? Hay chỉ chửi đổng vô cớ (Non-constructive)?",
            ),
            ARQQuery(
                id=3,
                question="Bước 3 - XÁC ĐỊNH ĐỐI TƯỢNG PHÀN NÀN: Có sự phàn nàn cụ thể về sản phẩm/dịch vụ/giao hàng không? (Lưu ý: Chê nhưng không mang tính xây dựng hoặc chửi bới nặng nề được coi là Non-complaint/Toxic).",
            ),
            ARQQuery(
                id=4,
                question=f"Bước 4 - QUYẾT ĐỊNH: Dựa trên định nghĩa (Label 1 = Constructive Complaint, Label 0 = Khen hoặc Chửi bới/Non-constructive), hãy chọn nhãn phù hợp nhất.",
            ),
        ]

    @staticmethod
    def _contextual_queries(text: str, context: Optional[str]) -> List[ARQQuery]:
        """Critic agent (formerly Contextual): Devil's Advocate analysis based on Olshtain & Weinbach.

        Focuses on verifying strictly against the definition constraints:
        - Complaint (1) MUST be constructive and show dissatisfaction/unmet expectation.
        - Non-complaint (0) includes compliments AND non-constructive hate speech/insults.
        """
        return [
            ARQQuery(
                id=1,
                question="Bước 1 - CRITIC CHECK (HATE SPEECH): Tìm kỹ các từ ngữ thô tục, xúc phạm, chửi thề (L**, c**, ngu, lừa đảo...). Theo định nghĩa, nếu chỉ có chửi bới mà KHÔNG mang tính xây dựng -> Phải là Label 0 (Non-complaint). Câu này có vi phạm không?",
            ),
            ARQQuery(
                id=2,
                question="Bước 2 - CRITIC CHECK (UNMET EXPECTATIONS): Tìm các cấu trúc 'Tuy nhiên', 'Giá mà', 'Phải chi', 'Nhưng', 'Hơi...'. Theo định nghĩa, Complaint (1) thường đi kèm mong muốn giải quyết hoặc sự thất vọng giữa kỳ vọng và thực tế. Câu này có chứa Wish/Suggestion ẩn sau lời khen không?",
            ),
            ARQQuery(
                id=3,
                question="Bước 3 - XÁC MINH TÍNH XÂY DỰNG (CONSTRUCTIVE): Complaint (1) phải mang tính xây dựng. Câu này có đưa ra vấn đề cụ thể để người bán cải thiện không? Hay chỉ là cảm xúc tiêu cực vô cớ?",
            ),
            ARQQuery(
                id=4,
                question="Bước 4 - PHÁN QUYẾT CUỐI CÙNG: Dựa trên 3 bước soi mói trên, hãy chọn nhãn. (Nhắc lại: Chửi bới thô tục = 0; Khen + Góp ý nhẹ = 1).",
            ),
        ]

    @staticmethod
    def _retrieval_queries(text: str) -> List[ARQQuery]:
        """Retrieval agent: analysis with similar examples."""
        return [
            ARQQuery(
                id=1,
                question="Bước 1 - PHÂN TÍCH VÍ DỤ: Các ví dụ tương tự được gán nhãn như thế nào? Chú ý các ví dụ có từ ngữ tiêu cực/chửi bới nhưng gán nhãn 0.",
            ),
            ARQQuery(
                id=2,
                question=f"Bước 2 - SO SÁNH TÍNH XÂY DỰNG: Comment '{text}' có mang tính xây dựng (constructive) giống các ví dụ Complaint (1) không? Hay giống các ví dụ Toxic/Non-complaint (0)?",
            ),
            ARQQuery(
                id=3,
                question="Bước 3 - PATTERN MATCHING: Có từ khóa chửi thề (profanity) hay xúc phạm nào xuất hiện? (Nếu có -> xu hướng về Label 0).",
            ),
            ARQQuery(
                id=4,
                question="Bước 4 - QUYẾT ĐỊNH: Dựa trên similarity, label nào phù hợp? (Label 1 = Complaint có tính xây dựng).",
            ),
        ]

    @staticmethod
    def _hybrid_queries(text: str) -> List[ARQQuery]:
        """Hybrid agent: analysis with edge cases and ambiguous patterns."""
        return [
            ARQQuery(
                id=1,
                question="Bước 1 - EDGE CASE ANALYSIS: Kiểm tra xem đây có phải là Edge Case: 'Chửi bới/Hateful' (Label 0) hoặc 'Khen nhưng thất vọng/Wish' (Label 1) không?",
            ),
            ARQQuery(
                id=2,
                question=f"Bước 2 - AMBIGUOUS CHECK: Comment '{text}' có mỉa mai (sarcasm) không? Có chửi thề (Profanity) không? (Chửi thề -> Non-complaint).",
            ),
            ARQQuery(
                id=3,
                question="Bước 3 - DECISION: Phân loại dứt khoát. Complaint (1) phải có ý định góp ý/phàn nàn cụ thể. Toxic/Hateful (0) chỉ là xả giận vô cớ.",
            ),
            ARQQuery(
                id=4,
                question="Bước 4 - REASONING: Giải thích tại sao chọn label này, đặc biệt nếu comment chứa từ ngữ tiêu cực.",
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
