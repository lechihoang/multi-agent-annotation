

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

    @staticmethod
    def _get_labels() -> Dict[str, str]:
        config = get_config()
        if (
            hasattr(config, "task")
            and hasattr(config.task, "labels")
            and isinstance(config.task.labels, dict)
        ):
            return config.task.labels

        return {
            "0": "Non-complaint - Khen ngợi thuần túy, hài lòng, tích cực.",
            "1": "Complaint - Phàn nàn, góp ý, không hài lòng về sản phẩm/dịch vụ/giao hàng.",
        }

    @staticmethod
    def _get_output_schema() -> Dict[str, Any]:
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
        queries = {
            "primary": ARQPromptBuilder._primary_queries(text),
            "contextual": ARQPromptBuilder._contextual_queries(text, context),
            "retrieval": ARQPromptBuilder._retrieval_queries(text),
            "hybrid": ARQPromptBuilder._hybrid_queries(text),
        }
        return queries.get(agent_type, queries["primary"])

    @staticmethod
    def _primary_queries(text: str) -> List[ARQQuery]:
        labels = ARQPromptBuilder._get_labels()
        return [
            ARQQuery(
                id=1,
                question="Bước 1 - PHÂN TÍCH NỘI DUNG: Comment thể hiện (a) hài lòng/khen, (b) không hài lòng/phàn nàn/góp ý, hay (c) chửi bới/xúc phạm/hate speech?",
            ),
            ARQQuery(
                id=2,
                question="Bước 2 - KIỂM TRA TÍNH XÂY DỰNG (Constructive): Nếu comment tiêu cực, nó có MANG TÍNH XÂY DỰNG không? Tức là: thể hiện kỳ vọng chưa được đáp ứng, góp ý cải thiện, cảnh báo, mong muốn (wish), gợi ý? Hay chỉ là chửi bới/xúc phạm/hate speech KHÔNG giúp doanh nghiệp cải thiện? (VD: 'Game như L** đừng tải' = xúc phạm, KHÔNG xây dựng → Label 0. 'Giao hàng chậm quá' = góp ý xây dựng → Label 1)",
            ),
            ARQQuery(
                id=3,
                question="Bước 3 - KIỂM TRA MIXED CONTENT: Comment có VỪA khen VỪA chê không? (VD: 'đẹp nhưng hơi nhỏ', 'tốt nhưng giao chậm'). Nếu có phàn nàn/góp ý mang tính xây dựng → Label 1.",
            ),
            ARQQuery(
                id=4,
                question="Bước 4 - QUYẾT ĐỊNH: Label 1 = Phàn nàn/góp ý/wish/cảnh báo MANG TÍNH XÂY DỰNG (kể cả mixed khen+chê). Label 0 = Khen ngợi thuần túy, hài lòng HOẶC chửi bới/xúc phạm/hate speech KHÔNG mang tính xây dựng.",
            ),
        ]

    @staticmethod
    def _contextual_queries(text: str, context: Optional[str]) -> List[ARQQuery]:
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
                question="Bước 3 - KIỂM TRA TÍNH XÂY DỰNG: Nếu comment tiêu cực, phân biệt: (a) Phàn nàn/góp ý mang tính xây dựng (kỳ vọng chưa đáp ứng, mong cải thiện) → Label 1. (b) Chửi bới/xúc phạm/hate speech KHÔNG xây dựng (VD: 'Game như L** đừng tải', 'ngu', 'rác') → Label 0. (c) Mixed khen+chê xây dựng → Label 1.",
            ),
            ARQQuery(
                id=4,
                question="Bước 4 - PHÁN QUYẾT: Label 1 = Phàn nàn/góp ý/wish MANG TÍNH XÂY DỰNG (kể cả mixed). Label 0 = Khen ngợi thuần túy HOẶC chửi bới/xúc phạm/hate speech KHÔNG mang tính xây dựng.",
            ),
        ]

    @staticmethod
    def _retrieval_queries(text: str) -> List[ARQQuery]:
        return [
            ARQQuery(
                id=1,
                question="Bước 1 - PHÂN TÍCH VÍ DỤ: Các ví dụ tương tự được gán nhãn như thế nào? Chú ý pattern: phàn nàn xây dựng → Label 1, khen thuần/chửi bới không xây dựng → Label 0.",
            ),
            ARQQuery(
                id=2,
                question=f"Bước 2 - SO SÁNH: Comment '{text}' giống ví dụ nào hơn? Ví dụ có phàn nàn/góp ý xây dựng (Label 1), ví dụ khen (Label 0), hay ví dụ chửi bới/xúc phạm không xây dựng (Label 0)?",
            ),
            ARQQuery(
                id=3,
                question="Bước 3 - KIỂM TRA TÍNH XÂY DỰNG: Comment có thể hiện kỳ vọng chưa đáp ứng, mong muốn cải thiện, góp ý cụ thể không? (→ Label 1). Hay chỉ là chửi bới/xúc phạm chung chung không giúp cải thiện? (→ Label 0).",
            ),
            ARQQuery(
                id=4,
                question="Bước 4 - QUYẾT ĐỊNH: Dựa trên similarity với ví dụ. Label 1 = Phàn nàn/góp ý MANG TÍNH XÂY DỰNG. Label 0 = Khen ngợi thuần túy HOẶC chửi bới/hate speech KHÔNG xây dựng.",
            ),
        ]

    @staticmethod
    def _hybrid_queries(text: str) -> List[ARQQuery]:
        return [
            ARQQuery(
                id=1,
                question="Bước 1 - EDGE CASE: Kiểm tra các trường hợp khó: (a) 'Khen nhưng có góp ý nhẹ' → Label 1, (b) 'Khen thuần túy' → Label 0, (c) 'Chửi bới/xúc phạm không xây dựng' (VD: 'Game như L** đừng tải', 'ngu', 'rác') → Label 0.",
            ),
            ARQQuery(
                id=2,
                question=f"Bước 2 - AMBIGUOUS CHECK: Comment '{text}' có ẩn ý phàn nàn không? (VD: 'được', 'tạm', 'cũng ok' có thể là khen miễn cưỡng). Phân biệt: phàn nàn xây dựng (kỳ vọng chưa đáp ứng) vs chửi bới/hate speech (không xây dựng).",
            ),
            ARQQuery(
                id=3,
                question="Bước 3 - DECISION: Label 1 = Phàn nàn/góp ý/wish MANG TÍNH XÂY DỰNG (kể cả mixed). Label 0 = Khen thuần túy HOẶC chửi bới/xúc phạm/hate speech KHÔNG xây dựng.",
            ),
            ARQQuery(
                id=4,
                question="Bước 4 - REASONING: Giải thích rõ tại sao chọn label. Nếu comment tiêu cực, giải thích nó xây dựng (→ Label 1) hay chỉ xúc phạm (→ Label 0).",
            ),
        ]

    @staticmethod
    def to_prompt(arq: ARQPrompt) -> str:
        parts = [arq.system_prompt]

        if arq.examples:
            parts.extend(["", "VÍ DỤ MINH HỌA:"])
            for i, ex in enumerate(arq.examples):
                parts.append(f'  - "{ex["text"]}" → Label: {ex["label"]}')

        parts.extend(
            [
                "",
                "HƯỚNG DẪN SUY LUẬN (trả lời theo các bước sau):",
            ]
        )
        for q in arq.reasoning_queries:
            parts.append(f"  {q.question}")

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
        response = response.strip()

        if response.startswith("```json"):
            response = response[7:-3]
        elif response.startswith("```"):
            response = response[3:-3]

        try:
            data = json.loads(response)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON response from LLM: {e}. Response: {response[:200]}"
            )

        if "final_label" not in data:
            raise ValueError(f"Missing 'final_label' in response: {data}")
        if "confidence" not in data:
            raise ValueError(f"Missing 'confidence' in response: {data}")
        if "reasoning" not in data:
            raise ValueError(f"Missing 'reasoning' in response: {data}")

        valid_labels = list(ARQPromptBuilder._get_labels().keys())
        if data["final_label"] not in valid_labels:
            raise ValueError(
                f"Invalid label: {data['final_label']}. Expected one of {valid_labels}"
            )

        try:
            conf_val = float(data["confidence"])
            if not (0.0 <= conf_val <= 1.0):
                raise ValueError(f"Confidence out of range: {conf_val}")

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
            score = ARQPromptBuilder.confidence_to_score(data["confidence"])
            return {
                "final_label": data["final_label"],
                "confidence": data["confidence"],
                "confidence_score": score,
                "reasoning": data["reasoning"],
            }

    @staticmethod
    def confidence_to_score(confidence: str) -> float:
        mapping = {
            "HIGH": 0.9,
            "MEDIUM": 0.7,
            "LOW": 0.5,
        }
        return mapping.get(confidence.upper(), 0.5)
