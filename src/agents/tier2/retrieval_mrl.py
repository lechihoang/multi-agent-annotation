

from typing import Dict, Any, List, Optional
import threading

from ...config import get_config, get_llm_client
from .arq_prompts import ARQPromptBuilder


class RetrievalMRLAnnotation:
    """Result from Retrieval-MRL agent."""

    def __init__(self, label: str, confidence: float, nearest_examples: List[Dict]):
        self.label = label
        self.confidence = confidence
        self.nearest_examples = nearest_examples

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "nearest_examples": self.nearest_examples,
        }


class RetrievalMrlAgent:

    _lock = threading.Lock()
    _model_loaded = False
    _embedding_model = None
    _faiss_index = None
    _examples: List[Dict] = []

    def __init__(self):
        self.config = get_config()
        self.weight = self.config.agents.hybrid
        self._init_unique_examples()
        self._llm_client = None
        self._init_llm()
        self._ensure_loaded()

    def _init_llm(self):
        try:
            self._llm_client = get_llm_client(self.config)
        except Exception as e:
            print(f"Warning: Could not initialize LLM client: {e}")
            self._llm_client = None

    def _ensure_loaded(self):
        if (
            RetrievalMrlAgent._model_loaded
            and RetrievalMrlAgent._faiss_index is not None
        ):
            return

        with RetrievalMrlAgent._lock:
            if (
                RetrievalMrlAgent._model_loaded
                and RetrievalMrlAgent._faiss_index is not None
            ):
                return

            try:
                from sentence_transformers import SentenceTransformer
                import faiss
                import numpy as np

                self.embedding_model = SentenceTransformer(
                    self.config.huggingface.embedding_model
                )

                examples = self._get_mrl_examples()
                texts = [ex["text"] for ex in examples]

                embeddings = self.embedding_model.encode(
                    texts, normalize_embeddings=True
                )

                dim = embeddings.shape[1]
                index = faiss.IndexFlatIP(dim)
                index.add(embeddings.astype("float32"))

                RetrievalMrlAgent._embedding_model = self.embedding_model
                RetrievalMrlAgent._faiss_index = index
                RetrievalMrlAgent._examples = examples
                RetrievalMrlAgent._model_loaded = True

            except ImportError as e:
                print(f"Warning: sentence-transformers or faiss not installed: {e}")
                RetrievalMrlAgent._embedding_model = None
                RetrievalMrlAgent._faiss_index = None

    def _init_unique_examples(self):
        self.examples = self._get_mrl_examples()

    def _get_mrl_examples(self) -> List[Dict]:
        seed_path = None
        if hasattr(self.config, "task") and hasattr(self.config.task, "paths"):
            seed_path = getattr(self.config.task.paths, "seed_file", None)
        elif hasattr(self.config, "paths"):
            seed_path = getattr(self.config.paths, "seed_file", None)

        if seed_path:
            try:
                import csv
                from pathlib import Path

                path = Path(seed_path)
                if not path.exists():
                    root_dir = Path(__file__).parent.parent.parent.parent
                    path = root_dir / seed_path

                if path.exists():
                    examples = []
                    with open(path, "r", encoding="utf-8-sig") as f:
                        reader = csv.DictReader(f)
                        cols = reader.fieldnames
                        text_col = next(
                            (
                                c
                                for c in cols
                                if c.lower() in ["comment", "text", "content", "review"]
                            ),
                            None,
                        )
                        label_col = next(
                            (
                                c
                                for c in cols
                                if c.lower() in ["label", "rating", "toxicity"]
                            ),
                            None,
                        )

                        if text_col and label_col:
                            for row in reader:
                                if row[text_col] and row[label_col]:
                                    examples.append(
                                        {
                                            "text": row[text_col].strip(),
                                            "label": str(row[label_col]).strip(),
                                        }
                                    )
                            return examples
                        else:
                            raise ValueError(
                                f"Could not identify text/label columns in {path}. "
                                f"Ensure columns match config or standard names."
                            )
            except Exception as e:
                print(f"Error: Failed to load seed examples: {e}")
                raise

        if not seed_path:
            raise ValueError("Seed file path not configured in config.yaml")

        raise ValueError(f"No valid examples found in seed file: {seed_path}")

    def _retrieve(self, text: str, k: int = 5) -> List[Dict]:
        if (
            RetrievalMrlAgent._faiss_index is None
            or RetrievalMrlAgent._embedding_model is None
        ):
            return []

        try:
            import numpy as np

            query_embedding = RetrievalMrlAgent._embedding_model.encode(
                [text], normalize_embeddings=True
            )

            distances, indices = RetrievalMrlAgent._faiss_index.search(
                query_embedding.astype("float32"), k
            )

            nearest = []
            for i, idx in enumerate(indices[0]):
                if 0 <= idx < len(RetrievalMrlAgent._examples):
                    similarity = float(distances[0][i])
                    nearest.append(
                        {
                            "text": RetrievalMrlAgent._examples[idx]["text"],
                            "label": RetrievalMrlAgent._examples[idx]["label"],
                            "similarity": similarity,
                        }
                    )

            return nearest

        except Exception:
            return []

    def _build_mafa_prompt(
        self, text: str, nearest: List[Dict], labels: List[str]
    ) -> str:
        self._ensure_loaded()

        arq_prompt = ARQPromptBuilder.build_toxicity_arq(
            text=text, examples=nearest, agent_type="hybrid"
        )
        return ARQPromptBuilder.to_prompt(arq_prompt)

    def _parse_response(self, content: str) -> Dict[str, Any]:
        result = ARQPromptBuilder.parse_response(content)

        return {
            "topic": result.get("final_label", "unknown"),
            "confidence": result.get("confidence_score", 0.5),
            "confidence_level": result.get("confidence", "MEDIUM"),
            "reasoning": result.get("reasoning", ""),
            "intent_analysis": result.get("reasoning", ""),
        }

    async def annotate(
        self, text: str, labels: List[str] | None = None
    ) -> RetrievalMRLAnnotation:
        if labels is None or len(labels) == 0:
            raise ValueError("Labels must be provided for RetrievalMrlAgent")

        self._ensure_loaded()

        if RetrievalMrlAgent._faiss_index is None or self._llm_client is None:
            raise RuntimeError(
                "RetrievalMrlAgent not fully initialized (FAISS or LLM missing)"
            )

        nearest = self._retrieve(text, k=3)

        if not nearest:
            raise ValueError("No similar examples found for retrieval")

        prompt = self._build_mafa_prompt(text, nearest, labels)

        response = await self._llm_client.chat([{"role": "user", "content": prompt}])
        result = self._parse_response(response.content)

        numeric_conf = result.get("confidence", 0.5)

        return RetrievalMRLAnnotation(
            label=result.get("topic", "unknown"),
            confidence=numeric_conf,
            nearest_examples=nearest,
        )

    def get_weight(self) -> float:
        return self.weight

    def unload(self):
        self._llm_client = None
