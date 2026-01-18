"""
Entity Extraction Agent.

Extracts named entities using Llama 70B models.
Tier 2 agent with weight 0.35.
"""

from typing import Any

import structlog

from ..config.settings import settings
from ..models import Annotation, ChatResponse, Entity, EntityAnnotation
from .base import BaseAnnotationAgent

logger = structlog.get_logger()


class EntityAgent(BaseAnnotationAgent):
    """
    Entity Extraction Agent.

    Uses Llama 70B (Groq/HuggingFace) for accurate entity extraction.
    Identifies PERSON, ORG, LOCATION, DATE, MONEY, and other entity types.

    Weight: 0.35 (core extraction task)
    """

    name = "entity_agent"
    weight = settings.entity_agent_weight
    tier = 2

    # Default entity types
    DEFAULT_ENTITY_TYPES = [
        "PERSON",
        "ORGANIZATION",
        "LOCATION",
        "DATE",
        "TIME",
        "MONEY",
        "PRODUCT",
        "EVENT",
        "EMAIL",
        "PHONE",
        "URL",
    ]

    def __init__(
        self,
        primary_provider: str = "groq",
        fallback_provider: str = "huggingface",
        primary_model: str = "llama-3.3-70b-versatile",
        fallback_model: str = "meta-llama/Llama-3.2-8B-Instruct",
        entity_types: list[str] | None = None,
    ):
        """
        Initialize Entity Agent.

        Args:
            primary_provider: Primary API provider
            fallback_provider: Fallback provider
            primary_model: Model for primary provider
            fallback_model: Model for fallback
            entity_types: Custom entity types to extract
        """
        super().__init__(
            primary_provider=primary_provider,
            fallback_provider=fallback_provider,
            primary_model=primary_model,
            fallback_model=fallback_model,
        )
        self.entity_types = entity_types or self.DEFAULT_ENTITY_TYPES

    def get_system_prompt(self) -> str:
        """Get system prompt for entity extraction."""
        prompts = self._prompts.get("entity_extraction", {})

        if "system" in prompts:
            return prompts["system"]

        types_str = ", ".join(self.entity_types)
        return f"""You are an expert named entity recognition (NER) agent. Your task is to extract
all relevant entities from the given text.

Entity types to identify: {types_str}

Guidelines:
- Extract exact text spans as they appear in the text
- Provide start and end character positions (0-indexed)
- Assign confidence scores for each entity
- Handle overlapping entities appropriately
- If no entities found, return empty list

Output format (JSON only):
{{
  "entities": [
    {{
      "text": "entity text",
      "type": "ENTITY_TYPE",
      "start": 0,
      "end": 10,
      "confidence": 0.95
    }}
  ],
  "confidence": 0.90
}}

IMPORTANT: Your response must be valid JSON only. Do not include any text before or after the JSON."""

    def get_user_prompt(self, text: str, **kwargs) -> str:
        """Get user prompt with text for entity extraction."""
        prompts = self._prompts.get("entity_extraction", {})

        if "user" in prompts:
            return prompts["user"].format(text=text)

        return f"""Extract all named entities from the following text:

Text: {text}

Provide your extraction in valid JSON format."""

    def parse_response(self, response: ChatResponse) -> dict[str, Any]:
        """Parse entity extraction response."""
        parsed = self._parse_json_response(response.content)

        # Extract and validate entities
        raw_entities = parsed.get("entities", [])
        entities = []

        for raw in raw_entities:
            if not isinstance(raw, dict):
                continue

            entity_type = raw.get("type", "UNKNOWN").upper()
            if entity_type not in self.entity_types:
                entity_type = "OTHER"

            entities.append({
                "text": raw.get("text", ""),
                "type": entity_type,
                "start": raw.get("start", 0),
                "end": raw.get("end", 0),
                "confidence": raw.get("confidence", 0.5),
            })

        # Overall confidence
        confidence = parsed.get("confidence", 0.5)
        if entities and not parsed.get("confidence"):
            # Calculate from entity confidences
            confidence = sum(e["confidence"] for e in entities) / len(entities)

        return {
            "entities": entities,
            "confidence": confidence,
            "entity_count": len(entities),
        }

    async def annotate(self, text: str, **kwargs) -> Annotation:
        """
        Extract entities from the given text.

        Args:
            text: Text to extract entities from
            **kwargs: Additional parameters

        Returns:
            Annotation with entity extraction result
        """
        messages = self._build_messages(text, **kwargs)

        response = await self._execute_with_fallback(
            messages=messages,
            temperature=0.0,  # More deterministic for extraction
            max_tokens=2048,
            json_mode=True,
        )

        result = self.parse_response(response)

        logger.info(
            "entities_extracted",
            entity_count=result.get("entity_count", 0),
            confidence=result.get("confidence"),
            latency_ms=round(response.latency_ms, 2),
        )

        return self._create_annotation(result, response.latency_ms)

    async def extract_entities(self, text: str) -> EntityAnnotation:
        """
        Convenience method that returns EntityAnnotation directly.

        Args:
            text: Text to extract entities from

        Returns:
            EntityAnnotation with extraction result
        """
        annotation = await self.annotate(text)
        result = annotation.result

        # Convert to Entity objects
        entities = [
            Entity(
                text=e["text"],
                type=e["type"],
                start=e["start"],
                end=e["end"],
                confidence=e["confidence"],
            )
            for e in result.get("entities", [])
        ]

        return EntityAnnotation(
            entities=entities,
            confidence=annotation.confidence,
        )

    def validate_positions(self, text: str, entities: list[dict]) -> list[dict]:
        """
        Validate and fix entity positions against the original text.

        Args:
            text: Original text
            entities: Extracted entities

        Returns:
            Entities with validated/fixed positions
        """
        validated = []

        for entity in entities:
            entity_text = entity.get("text", "")
            start = entity.get("start", 0)
            end = entity.get("end", 0)

            # Try to find the entity text in the original
            if text[start:end] == entity_text:
                validated.append(entity)
            else:
                # Try to find correct position
                idx = text.find(entity_text)
                if idx >= 0:
                    entity["start"] = idx
                    entity["end"] = idx + len(entity_text)
                    validated.append(entity)
                else:
                    # Case-insensitive search
                    idx = text.lower().find(entity_text.lower())
                    if idx >= 0:
                        entity["start"] = idx
                        entity["end"] = idx + len(entity_text)
                        entity["text"] = text[idx : idx + len(entity_text)]
                        validated.append(entity)
                    else:
                        logger.warning(
                            "entity_not_found_in_text",
                            entity_text=entity_text,
                        )

        return validated


# Factory function
def create_entity_agent(**kwargs) -> EntityAgent:
    """Create an EntityAgent with optional configuration."""
    return EntityAgent(**kwargs)
