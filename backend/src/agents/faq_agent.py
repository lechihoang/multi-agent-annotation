"""
FAQ/Pattern Matching Agent.

Matches user queries against known FAQ patterns.
Tier 2 agent with weight 0.30.
"""

from typing import Any

import structlog

from ..config.settings import settings
from ..models import Annotation, ChatResponse, FAQAnnotation
from .base import BaseAnnotationAgent

logger = structlog.get_logger()


class FAQAgent(BaseAnnotationAgent):
    """
    FAQ/Pattern Matching Agent.

    Uses Llama 8B (HuggingFace/Ollama) for pattern matching against FAQ database.
    Lightweight model suitable for high-volume, low-complexity matching.

    Weight: 0.30 (pattern matching support)
    """

    name = "faq_agent"
    weight = settings.faq_agent_weight
    tier = 2

    # Default FAQ categories
    DEFAULT_CATEGORIES = [
        "account",
        "billing",
        "technical",
        "product",
        "shipping",
        "returns",
        "general",
    ]

    # Sample FAQ database (in production, load from external source)
    DEFAULT_FAQ_DATABASE = {
        "account": [
            "How do I create an account?",
            "How do I reset my password?",
            "How do I update my profile?",
            "How do I delete my account?",
        ],
        "billing": [
            "What payment methods do you accept?",
            "How do I update my payment information?",
            "Where can I find my invoice?",
            "How do I cancel my subscription?",
        ],
        "technical": [
            "The app is not loading",
            "I'm getting an error message",
            "How do I clear my cache?",
            "The feature is not working",
        ],
        "product": [
            "What features are included?",
            "Is there a free trial?",
            "What are the pricing plans?",
            "Do you offer enterprise solutions?",
        ],
        "shipping": [
            "How long does shipping take?",
            "Do you ship internationally?",
            "How do I track my order?",
            "What are the shipping costs?",
        ],
        "returns": [
            "What is your return policy?",
            "How do I return an item?",
            "How long do refunds take?",
            "Can I exchange an item?",
        ],
        "general": [
            "How do I contact support?",
            "What are your business hours?",
            "Where are you located?",
            "Do you have a mobile app?",
        ],
    }

    def __init__(
        self,
        primary_provider: str = "huggingface",
        fallback_provider: str = "ollama",
        primary_model: str = "meta-llama/Llama-3.2-8B-Instruct",
        fallback_model: str = "llama3.2",
        faq_database: dict[str, list[str]] | None = None,
        categories: list[str] | None = None,
    ):
        """
        Initialize FAQ Agent.

        Args:
            primary_provider: Primary API provider
            fallback_provider: Fallback provider
            primary_model: Model for primary provider
            fallback_model: Model for fallback
            faq_database: Custom FAQ database
            categories: Custom categories
        """
        super().__init__(
            primary_provider=primary_provider,
            fallback_provider=fallback_provider,
            primary_model=primary_model,
            fallback_model=fallback_model,
        )
        self.faq_database = faq_database or self.DEFAULT_FAQ_DATABASE
        self.categories = categories or self.DEFAULT_CATEGORIES

    def get_system_prompt(self) -> str:
        """Get system prompt for FAQ matching."""
        prompts = self._prompts.get("faq_matching", {})

        if "system" in prompts:
            return prompts["system"]

        categories_str = ", ".join(self.categories)
        return f"""You are an FAQ matching agent. Your task is to determine if the given text
matches any known FAQ patterns or categories.

Available categories: {categories_str}

Guidelines:
- Compare semantic meaning, not just keywords
- Calculate similarity scores based on meaning
- Identify the most relevant FAQ category
- If no good match exists (similarity < 0.5), set matched_faq to null
- Be conservative with confidence scores

Output format (JSON only):
{{
  "matched_faq": "The most similar FAQ question or null if no match",
  "similarity_score": 0.85,
  "confidence": 0.90,
  "category": "faq_category"
}}

IMPORTANT: Your response must be valid JSON only."""

    def get_user_prompt(self, text: str, **kwargs) -> str:
        """Get user prompt with FAQ context."""
        prompts = self._prompts.get("faq_matching", {})

        # Build FAQ context
        faq_context = []
        for category, questions in self.faq_database.items():
            faq_context.append(f"\n{category.upper()}:")
            for q in questions:
                faq_context.append(f"  - {q}")
        faq_str = "\n".join(faq_context)

        if "user" in prompts:
            return prompts["user"].format(
                text=text,
                faq_categories=faq_str,
            )

        return f"""Match the following text against known FAQ patterns:

Text: {text}

FAQ Database:
{faq_str}

Provide your matching result in valid JSON format."""

    def parse_response(self, response: ChatResponse) -> dict[str, Any]:
        """Parse FAQ matching response."""
        parsed = self._parse_json_response(response.content)

        matched_faq = parsed.get("matched_faq")
        similarity_score = parsed.get("similarity_score", 0.0)
        confidence = parsed.get("confidence", 0.5)
        category = parsed.get("category", "general")

        # Normalize category
        if isinstance(category, str):
            category = category.lower().strip()
            if category not in self.categories:
                category = "general"

        # Handle null/None matched_faq
        if matched_faq in [None, "null", "None", ""]:
            matched_faq = None
            similarity_score = 0.0

        return {
            "matched_faq": matched_faq,
            "similarity_score": similarity_score,
            "confidence": confidence,
            "category": category,
            "has_match": matched_faq is not None,
        }

    async def annotate(self, text: str, **kwargs) -> Annotation:
        """
        Match text against FAQ patterns.

        Args:
            text: Text to match
            **kwargs: Additional parameters

        Returns:
            Annotation with FAQ matching result
        """
        messages = self._build_messages(text, **kwargs)

        response = await self._execute_with_fallback(
            messages=messages,
            temperature=0.1,
            max_tokens=1024,
            json_mode=True,
        )

        result = self.parse_response(response)

        logger.info(
            "faq_matched",
            has_match=result.get("has_match"),
            category=result.get("category"),
            similarity=result.get("similarity_score"),
            confidence=result.get("confidence"),
            latency_ms=round(response.latency_ms, 2),
        )

        return self._create_annotation(result, response.latency_ms)

    async def match_faq(self, text: str) -> FAQAnnotation:
        """
        Convenience method that returns FAQAnnotation directly.

        Args:
            text: Text to match

        Returns:
            FAQAnnotation with matching result
        """
        annotation = await self.annotate(text)
        result = annotation.result

        return FAQAnnotation(
            matched_faq=result.get("matched_faq"),
            similarity_score=result.get("similarity_score", 0.0),
            confidence=annotation.confidence,
            category=result.get("category"),
        )

    def add_faq(self, category: str, question: str) -> None:
        """
        Add a new FAQ question to the database.

        Args:
            category: FAQ category
            question: FAQ question
        """
        if category not in self.faq_database:
            self.faq_database[category] = []
            if category not in self.categories:
                self.categories.append(category)

        if question not in self.faq_database[category]:
            self.faq_database[category].append(question)

    def load_faq_database(self, faq_data: dict[str, list[str]]) -> None:
        """
        Load FAQ database from external source.

        Args:
            faq_data: Dictionary of category -> questions
        """
        self.faq_database = faq_data
        self.categories = list(faq_data.keys())
        logger.info(
            "faq_database_loaded",
            categories=len(self.categories),
            total_questions=sum(len(q) for q in faq_data.values()),
        )


# Factory function
def create_faq_agent(**kwargs) -> FAQAgent:
    """Create a FAQAgent with optional configuration."""
    return FAQAgent(**kwargs)
