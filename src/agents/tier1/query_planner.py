"""Query Planner Agent - MAFA Tier 1 Component.

Implements MAFA's Query Planning Agent (Section 4.2):
- Powered by LLM (Groq)
- Two-stage: intent analysis + contextual expansion
- Caching for production performance

Flow:
  Input: "tôi thấy người lái xe hơi bấm còi"
    ↓
  QueryPlanner: Intent analysis + Expansion
    ↓
  Output: "comment về giao thông, phương tiện, vi phạm giao thông"
    ↓
  4 Agents process expanded query
"""

from typing import Dict, Any, Optional
import json


class QueryPlanner:
    """MAFA Query Planning Agent.

    Analyzes and expands user queries for better annotation quality.

    Usage:
        planner = QueryPlanner(groq_client)
        expanded = planner.expand("user comment here")
    """

    def __init__(self, llm_client):
        """Initialize query planner with LLM client."""
        self._llm_client = llm_client
        self._cache: Dict[str, str] = {}

    def _build_prompt(self, query: str) -> str:
        """Build MAFA-style ARQ prompt for query planning.

        Dynamic prompt based on general intent analysis instructions.
        """
        return f"""Bạn là chuyên gia phân tích và mở rộng query cho hệ thống phân loại văn bản.

QUERY: {query}

NHIỆM VỤ:
1. Phân tích intent chính của comment
2. Mở rộng query với các từ khóa liên quan (đồng nghĩa, ngữ cảnh)
3. Trả về expanded query để giúp search engine hoặc model hiểu rõ ngữ nghĩa

QUY TẮC:
- Giữ nguyên ý nghĩa gốc
- Thêm các từ khóa làm rõ ngữ cảnh (ví dụ: nếu khen sản phẩm -> thêm "tích cực", "hài lòng")
- Không thay đổi quan điểm của người viết

Respond JSON:
{{"intent": "...", "expanded_query": "..."}}"""

    def _parse_response(self, content: str) -> Dict[str, str]:
        """Parse JSON from LLM response."""
        content = content.strip()

        if content.startswith("```json"):
            content = content[7:-3]
        elif content.startswith("```"):
            content = content[3:-3]

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            import re

            json_match = re.search(r"\{[^{}]*\}", content)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
            return {"intent": "unknown", "expanded_query": content}

    async def expand(self, query: str, use_cache: bool = True) -> str:
        """Expand query using LLM.

        Args:
            query: Original user query
            use_cache: Use cached results (default: True)

        Returns:
            Expanded query for better annotation quality
        """
        # Check cache
        if use_cache and query in self._cache:
            return self._cache[query]

        try:
            prompt = self._build_prompt(query)
            response = await self._llm_client.chat(
                [{"role": "user", "content": prompt}]
            )
            result = self._parse_response(response.content)

            expanded = result.get("expanded_query", query)

            # Cache result
            if use_cache:
                self._cache[query] = expanded

            return expanded

        except Exception as e:
            print(f"Error: Query planning failed: {e}")
            raise  # NO FALLBACK - Propagate error

    async def expand_batch(self, queries: list, use_cache: bool = True) -> list:
        """Expand multiple queries in parallel."""
        import asyncio

        return await asyncio.gather(*[self.expand(q, use_cache) for q in queries])

    def clear_cache(self):
        """Clear query cache."""
        self._cache.clear()

    def cache_size(self) -> int:
        """Get cache size."""
        return len(self._cache)

    def get_expansion_info(self, query: str) -> Dict:
        """Get detailed expansion info (requires LLM call)."""
        import asyncio

        class SyncWrapper:
            def __init__(self, planner):
                self.planner = planner

        # This is async-only, so we need to handle it
        return {"query": query, "note": "Use expand() for async access"}
