"""Few-shot example selection using Groq API."""

from typing import List, Dict, Any, Set, Optional
import threading
import random
import hashlib
import json
import csv


class FewShotExample:
    """Single few-shot example."""

    def __init__(
        self, text: str, label: str, metadata: Optional[Dict[str, Any]] = None
    ):
        self.text = text
        self.label = label
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {"text": self.text, "label": self.label, "metadata": self.metadata}


class FewShotSelector:
    """Selects unique few-shot examples for MAFA agents using Groq API.

    Each agent gets different examples to increase ensemble diversity.
    """

    _lock = threading.Lock()
    _data_loaded = False
    _all_examples: List[FewShotExample] = []

    AGENT_CONFIGS = {
        "primary_only": {"diversity": 0.3, "min_len": 20, "max_len": 100},
        "contextual": {"diversity": 0.5, "min_len": 50, "prefer_title": True},
        "retrieval": {"diversity": 0.8, "min_len": 0, "prefer_title": False},
        "retrieval_mrl": {"diversity": 0.9, "min_len": 0, "prefer_title": False},
    }

    def __init__(self, training_data_path: str, llm_client=None, cache_size: int = 500):
        self.training_data_path = training_data_path
        self.llm_client = llm_client
        self.cache_size = cache_size
        self._selection_cache: Dict[str, List[FewShotExample]] = {}
        self._used_texts: Set[str] = set()
        self._ensure_loaded()

    def _ensure_loaded(self):
        if FewShotSelector._data_loaded:
            return
        with FewShotSelector._lock:
            if FewShotSelector._data_loaded:
                return
            try:
                examples = []
                with open(self.training_data_path, "r", encoding="utf-8-sig") as f:
                    reader = csv.DictReader(f)
                    cols = reader.fieldnames or []
                    # Flexible column matching (like RetrievalAgent)
                    text_col = next(
                        (c for c in cols if c.lower() in ["comment", "text", "content", "review"]),
                        None,
                    )
                    label_col = next(
                        (c for c in cols if c.lower() in ["label", "rating", "toxicity"]),
                        None,
                    )
                    title_col = next(
                        (c for c in cols if c.lower() in ["title"]),
                        None,
                    )
                    topic_col = next(
                        (c for c in cols if c.lower() in ["topic", "domain"]),
                        None,
                    )
                    if not text_col or not label_col:
                        raise ValueError(
                            f"Could not identify text/label columns in {self.training_data_path}. "
                            f"Found columns: {cols}. Expected text column (comment/text/content/review) "
                            f"and label column (label/rating/toxicity)."
                        )
                    for row in reader:
                        comment = row.get(text_col, "").strip()
                        label_val = row.get(label_col, "").strip()
                        # Normalize float labels like "0.0"/"1.0" to "0"/"1"
                        if label_val in ["0.0", "1.0"]:
                            label_val = label_val[0]
                        if comment and label_val in ["0", "1"]:
                            examples.append(
                                FewShotExample(
                                    text=comment,
                                    label=label_val,
                                    metadata={
                                        "title": row.get(title_col, "").strip() if title_col else "",
                                        "topic": row.get(topic_col, "").strip() if topic_col else "",
                                    },
                                )
                            )
                FewShotSelector._all_examples = examples
                FewShotSelector._data_loaded = True
                print(f"✓ Loaded {len(examples)} examples")
            except Exception as e:
                print(f"Warning: Failed to load data: {e}")
                FewShotSelector._all_examples = []

    def _get_cache_key(self, query: str, agent: str, k: int) -> str:
        key = f"{query[:30]}:{agent}:{k}"
        return hashlib.md5(key.encode()).hexdigest()

    def _filter_by_length(
        self,
        examples: List[FewShotExample],
        min_len: int,
        max_len: Optional[int] = None,
    ) -> List[FewShotExample]:
        max_val = max_len if max_len is not None else float("inf")
        return [ex for ex in examples if min_len <= len(ex.text) <= max_val]

    def _select_balanced(self, k: int, exclude: Set[str]) -> List[FewShotExample]:
        selected = []
        k0, k1 = max(1, k // 2), k - max(1, k // 2)

        available_0 = [
            ex
            for ex in FewShotSelector._all_examples
            if ex.label == "0" and ex.text not in exclude
        ]
        available_1 = [
            ex
            for ex in FewShotSelector._all_examples
            if ex.label == "1" and ex.text not in exclude
        ]

        if available_0:
            selected.extend(random.sample(available_0, min(k0, len(available_0))))
        if available_1:
            selected.extend(random.sample(available_1, min(k1, len(available_1))))

        random.shuffle(selected)
        return selected[:k]

    def _select_edge_cases(self, k: int, exclude: Set[str]) -> List[FewShotExample]:
        edge_cases = []
        for ex in FewShotSelector._all_examples:
            if ex.text in exclude:
                continue
            text_len = len(ex.text)
            is_short = text_len < 30
            has_irony = any(
                w in ex.text.lower() for w in ["thật", "quá", "hay", "tuyệt"]
            )

            if (
                (ex.label == "1" and is_short)
                or (ex.label == "0" and is_short and has_irony)
                or (ex.label == "1" and text_len > 200)
            ):
                edge_cases.append(ex)

        if len(edge_cases) > k:
            return random.sample(edge_cases, k)
        return edge_cases[:k]

    async def _select_with_llm(
        self, query: str, agent: str, k: int, exclude: Set[str]
    ) -> List[FewShotExample]:
        """Use LLM to intelligently select examples based on query."""
        if not self.llm_client:
            return self._select_fallback(agent, k, exclude)

        exclude_list = list(exclude)[:20]
        prompt = f"""Select {k} diverse examples for annotating: "{query}"

Rules:
- Agent: {agent}
- Exclude: {exclude_list}
- Provide balanced examples

Return ONLY JSON array:
[{{"text": "...", "label": "0/1", "reason": "..."}}]"""

        try:
            response = await self.llm_client.chat([{"role": "user", "content": prompt}])
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:-3]

            selected_data = json.loads(content)
            selected = []
            for item in selected_data:
                ex = FewShotExample(
                    text=item.get("text", ""),
                    label=item.get("label", "0"),
                    metadata={"reason": item.get("reason", "")},
                )
                if ex.text and ex.text not in exclude:
                    selected.append(ex)
            return selected[:k]
        except Exception:
            return self._select_fallback(agent, k, exclude)

    def _select_fallback(
        self, agent: str, k: int, exclude: Set[str]
    ) -> List[FewShotExample]:
        config = self.AGENT_CONFIGS.get(agent, self.AGENT_CONFIGS["retrieval"])
        examples = [
            ex for ex in FewShotSelector._all_examples if ex.text not in exclude
        ]

        if config["min_len"] > 0:
            examples = self._filter_by_length(
                examples, config["min_len"], config.get("max_len")
            )

        if len(examples) >= k:
            return random.sample(examples, k)

        all_available = [
            ex for ex in FewShotSelector._all_examples if ex.text not in exclude
        ]
        return random.sample(all_available, min(k, len(all_available)))

    def select_examples(
        self,
        query: str,
        agent_name: str,
        task_type: str = "toxicity",
        k: int = 8,
        use_cache: bool = True,
        exclude_texts: Optional[Set[str]] = None,
    ) -> List[FewShotExample]:
        exclude = exclude_texts or set()
        cache_key = self._get_cache_key(query, agent_name, k)

        if use_cache and cache_key in self._selection_cache:
            cached = [
                ex for ex in self._selection_cache[cache_key] if ex.text not in exclude
            ]
            if len(cached) >= k // 2:
                return cached[:k]

        import asyncio

        if self.llm_client:
            examples = asyncio.run(self._select_with_llm(query, agent_name, k, exclude))
        else:
            examples = self._select_fallback(agent_name, k, exclude)

        if use_cache:
            self._selection_cache[cache_key] = examples
            if len(self._selection_cache) > self.cache_size:
                keys_to_remove = list(self._selection_cache.keys())[: -self.cache_size]
                for key in keys_to_remove:
                    del self._selection_cache[key]

        return examples[:k]

    def select_diverse_for_all_agents(
        self, query: str, task_type: str = "toxicity", k_per_agent: int = 8
    ) -> Dict[str, List[FewShotExample]]:
        """Select unique examples for all 4 agents (no overlap)."""
        agents = ["primary_only", "contextual", "retrieval", "retrieval_mrl"]
        result = {}
        used: Set[str] = set()

        for agent in agents:
            examples = self.select_examples(
                query=query,
                agent_name=agent,
                task_type=task_type,
                k=k_per_agent,
                use_cache=False,
                exclude_texts=used,
            )
            for ex in examples:
                used.add(ex.text)
            result[agent] = examples

        return result

    def stats(self) -> Dict[str, Any]:
        return {
            "total_examples": len(FewShotSelector._all_examples),
            "cache_size": len(self._selection_cache),
            "agents": list(self.AGENT_CONFIGS.keys()),
        }

    def clear_cache(self):
        self._selection_cache.clear()
        self._used_texts.clear()
