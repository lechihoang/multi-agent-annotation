"""Batch Processing Module - Send 50 samples in 1 prompt.

Efficient batch inference: 1 API call = 50 samples labeled.
"""

import json
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime


class BatchProcessor:
    """Process multiple samples in a single API call."""

    def __init__(self, llm_client, batch_size: int = 50):
        """Initialize batch processor.

        Args:
            llm_client: LLM client (Groq or NIM)
            batch_size: Number of samples per batch (default: 50)
        """
        self.client = llm_client
        self.batch_size = batch_size

    def _build_batch_prompt(
        self,
        texts: List[str],
        labels: List[str],
        is_toxicity: bool = True,
        few_shot_examples: Optional[List[Dict]] = None,
    ) -> str:
        """Build batch prompt with all samples.

        Args:
            texts: List of texts to classify
            labels: Classification labels
            is_toxicity: True for toxicity (0/1), False for topic classification
            few_shot_examples: Optional examples
        """
        labels_str = ", ".join(labels)

        # Build few-shot section
        examples_section = ""
        if few_shot_examples:
            examples_section = "\n\nVÍ DỤ MINH HỌA:\n"
            for ex in few_shot_examples[:5]:
                examples_section += f'- "{ex["text"]}" → Label: {ex["label"]}\n'

        if is_toxicity:
            return f"""Bạn là chuyên gia phân loại toxicity cho comment tiếng Việt.

TOXICITY LABELS:
- 0: Không toxic (bình thường, tích cực, hoặc chỉ bình luận thông thường)
- 1: Toxic (có lời lẽ xúc phạm, chửi rủa, phân biệt đối xử, đe dọa, hoặc gây thù ghét)

QUY TẮC:
- Comment tích cực hoặc trung tính → 0
- Comment có lời lẽ xúc phạm, chửi rủa, phân biệt đối xử → 1
{examples_section}
DANH SÁCH CẦN PHÂN LOẠI:
"""
        else:
            return f"""You are a topic classification expert. Classify the topic of each text.

TOPICS: {labels_str}

INSTRUCTIONS:
1. Classify each text into ONE of the provided topics
2. Provide confidence score (0.0-1.0)
3. Provide brief reasoning
{examples_section}
DANH SÁCH CẦN PHÂN LOẠI:
"""

    def _parse_batch_response(
        self, content: str, num_samples: int
    ) -> List[Dict[str, Any]]:
        """Parse batch response from LLM.

        Args:
            content: Raw response from LLM
            num_samples: Expected number of samples

        Returns:
            List of annotation results
        """
        content = content.strip()

        # Try to extract JSON array
        if content.startswith("```json"):
            content = content[7:-3]
        elif content.startswith("```"):
            content = content[3:-3]

        try:
            data = json.loads(content)
            if isinstance(data, list):
                results = []
                for item in data:
                    # Handle both string and integer labels
                    label = str(item.get("label", "unknown"))
                    results.append(
                        {
                            "label": label,
                            "confidence": float(item.get("confidence", 0.5)),
                            "reasoning": str(item.get("reasoning", "")),
                        }
                    )
                return results
            elif isinstance(data, dict) and "results" in data:
                return data["results"]
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            print(f"  [DEBUG] JSON parse error: {e}, trying regex...")

        # Fallback: try regex to find labels
        import re

        results = []
        # Look for patterns like "label": 1 or "label": "1"
        label_pattern = r'"index"\s*:\s*\d+.*?"label"\s*:\s*["\']?(\d+)["\']?'
        matches = re.findall(label_pattern, content, re.DOTALL)

        if matches:
            for label in matches[:num_samples]:
                results.append(
                    {
                        "label": str(label),
                        "confidence": 0.5,
                        "reasoning": "Regex fallback",
                    }
                )
        else:
            for i in range(num_samples):
                results.append(
                    {
                        "label": "unknown",
                        "confidence": 0.5,
                        "reasoning": "Parse error",
                    }
                )
        return results

    async def annotate_batch(
        self,
        texts: List[str],
        labels: List[str],
        few_shot_examples: Optional[List[Dict]] = None,
    ) -> List[Dict[str, Any]]:
        """Annotate a batch of texts in a single API call.

        Args:
            texts: List of texts to classify
            labels: Classification labels
            few_shot_examples: Optional few-shot examples

        Returns:
            List of annotation results (label, confidence, reasoning)
        """
        if not texts:
            return []

        # Build prompt
        is_toxicity = set(labels) == {"0", "1"}
        prompt = self._build_batch_prompt(texts, labels, is_toxicity, few_shot_examples)

        # Add each text with index
        for i, text in enumerate(texts):
            prompt += f'{i + 1}. "{text}"\n'

        prompt += """
YÊU CẦU:
- Trả về JSON array với format:
[{"index": 1, "label": "0 hoặc 1", "confidence": 0.0-1.0, "reasoning": "..."}, ...]
- Chỉ trả về JSON, không giải thích thêm.
"""

        try:
            response = await self.client.chat([{"role": "user", "content": prompt}])

            results = self._parse_batch_response(response.content, len(texts))

            # Ensure results have correct structure
            for i, result in enumerate(results):
                if "label" not in result:
                    result["label"] = "unknown"
                if "confidence" not in result:
                    result["confidence"] = 0.5
                if "reasoning" not in result:
                    result["reasoning"] = ""

            return results

        except Exception as e:
            print(f"Batch annotation error: {e}")
            # Return error results
            return [
                {"label": "unknown", "confidence": 0.5, "reasoning": f"Error: {str(e)}"}
                for _ in texts
            ]


async def process_batch(
    llm_client,
    texts: List[str],
    labels: List[str] = None,
    batch_size: int = 50,
) -> List[Dict[str, Any]]:
    """Process texts in batches.

    Args:
        llm_client: LLM client
        texts: All texts to process
        labels: Classification labels (auto-detected if None)
        batch_size: Samples per batch

    Returns:
        All annotation results
    """
    if labels is None:
        labels = ["0", "1"]

    processor = BatchProcessor(llm_client, batch_size)
    all_results = []

    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(texts) + batch_size - 1) // batch_size

        print(
            f"  Processing batch {batch_num}/{total_batches} ({len(batch)} samples)..."
        )

        results = await processor.annotate_batch(batch, labels)
        all_results.extend(results)

    return all_results
