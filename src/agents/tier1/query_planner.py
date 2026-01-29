

from typing import Dict, Any, Optional
import json


class QueryPlanner:

    def __init__(self, llm_client):
        self._llm_client = llm_client
        self._cache: Dict[str, str] = {}

    def _build_prompt(self, query: str) -> str:
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
        if use_cache and query in self._cache:
            return self._cache[query]

        try:
            prompt = self._build_prompt(query)
            response = await self._llm_client.chat(
                [{"role": "user", "content": prompt}]
            )
            result = self._parse_response(response.content)

            expanded = result.get("expanded_query", query)

            if use_cache:
                self._cache[query] = expanded

            return expanded

        except Exception as e:
            print(f"Error: Query planning failed: {e}")
            raise

    async def expand_batch(self, queries: list, use_cache: bool = True) -> list:
        import asyncio

        return await asyncio.gather(*[self.expand(q, use_cache) for q in queries])

    def clear_cache(self):
        self._cache.clear()

    def cache_size(self) -> int:
        return len(self._cache)

    def get_expansion_info(self, query: str) -> Dict:
        import asyncio

        class SyncWrapper:
            def __init__(self, planner):
                self.planner = planner

        return {"query": query, "note": "Use expand() for async access"}
