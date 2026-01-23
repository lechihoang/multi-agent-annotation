"""Retrieval-based Annotation Agent - MAFA Tier 2 Agent 3.

Agent C: Embedding-based retrieval with sentence-transformers and FAISS.
Uses dense vector representations for similarity search, then LLM for final classification
following MAFA's ARQ-style structured prompting with systematic reasoning steps.

MAFA Section 4.2.1: ARQ prompts with retrieval-augmented reasoning.
"""

from typing import Dict, Any, List, Optional
import threading

from ...config import get_config, get_llm_client
from .arq_prompts import ARQPromptBuilder


class RetrievalAnnotation:
    """Result from retrieval-based annotation."""

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


class RetrievalAgent:
    """MAFA Agent 3: Embedding-Enhanced Ranker.

    - Uses sentence-transformers for embeddings (all-MiniLM-L6-v2)
    - FAISS for approximate nearest neighbor search
    - LLM for final classification with structured ARQ-style prompting
    - Weight: 0.25 in MAFA ensemble
    """

    _lock = threading.Lock()
    _model_loaded = False
    _embedding_model = None
    _faiss_index = None
    _examples: List[Dict] = []

    def __init__(self):
        self.config = get_config()
        self.weight = self.config.agents.retrieval
        self._llm_client = None
        self._init_llm()

    def _init_llm(self):
        """Initialize LLM client (Groq or NVIDIA) based on config."""
        try:
            self._llm_client = get_llm_client(self.config)
        except Exception as e:
            print(f"Warning: Could not initialize LLM client: {e}")
            self._llm_client = None

    def _ensure_loaded(self):
        """Ensure model and index are loaded (thread-safe lazy loading)."""
        if RetrievalAgent._model_loaded and RetrievalAgent._faiss_index is not None:
            return

        with RetrievalAgent._lock:
            if RetrievalAgent._model_loaded and RetrievalAgent._faiss_index is not None:
                return

            try:
                from sentence_transformers import SentenceTransformer
                import faiss
                import numpy as np

                # Load embedding model
                self.embedding_model = SentenceTransformer(
                    self.config.huggingface.embedding_model
                )

                # Get topic examples (Might raise error if no seed data)
                examples = self._get_topic_examples()
                texts = [ex["text"] for ex in examples]

                # Encode all examples
                embeddings = self.embedding_model.encode(
                    texts, normalize_embeddings=True
                )

                # Create FAISS index (Inner Product for cosine similarity with normalized vectors)
                dim = embeddings.shape[1]
                index = faiss.IndexFlatIP(dim)
                index.add(embeddings.astype("float32"))

                RetrievalAgent._embedding_model = self.embedding_model
                RetrievalAgent._faiss_index = index
                RetrievalAgent._examples = examples
                RetrievalAgent._model_loaded = True

            except ImportError as e:
                print(f"Warning: sentence-transformers or faiss not installed: {e}")
                RetrievalAgent._embedding_model = None
                RetrievalAgent._faiss_index = None
            except Exception as e:
                print(f"Warning: Failed to load FAISS model: {e}")
                RetrievalAgent._embedding_model = None
                RetrievalAgent._faiss_index = None

    def _get_topic_examples(self) -> List[Dict]:
        """Get labeled examples for retrieval.

        Loads from 'seed_file' defined in config if available.
        Otherwise falls back to empty list (should be handled by caller).
        """
        seed_path = None
        if hasattr(self.config, "task") and hasattr(self.config.task, "paths"):
            seed_path = getattr(self.config.task.paths, "seed_file", None)
        elif hasattr(self.config, "paths"):  # Fallback for flat config
            seed_path = getattr(self.config.paths, "seed_file", None)

        if seed_path:
            try:
                import csv
                from pathlib import Path

                path = Path(seed_path)
                if not path.exists():
                    # Try relative to project root if not absolute
                    root_dir = Path(__file__).parent.parent.parent.parent
                    path = root_dir / seed_path

                if path.exists():
                    examples = []
                    with open(path, "r", encoding="utf-8-sig") as f:
                        reader = csv.DictReader(f)
                        # Normalize column names (case insensitive)
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
                            print(
                                f"Warning: Could not identify text/label columns in {path}. Using defaults."
                            )
            except Exception as e:
                print(f"Error: Failed to load seed examples from {seed_path}: {e}")
                # Re-raise to prevent silent failure if file is corrupt
                raise

        # NO FALLBACK - If we reach here, it means no seed path configured or file not found/empty
        if not seed_path:
            raise ValueError(
                "Seed file path not configured in config.yaml (task.paths.seed_file)"
            )

        raise ValueError(f"No valid examples found in seed file: {seed_path}")

    def _retrieve(self, text: str, k: int = 5) -> List[Dict]:
        """Retrieve most similar examples."""
        if (
            RetrievalAgent._faiss_index is None
            or RetrievalAgent._embedding_model is None
        ):
            return []

        try:
            import numpy as np

            # Encode query
            query_embedding = RetrievalAgent._embedding_model.encode(
                [text], normalize_embeddings=True
            )

            # Search FAISS
            distances, indices = RetrievalAgent._faiss_index.search(
                query_embedding.astype("float32"), k
            )

            nearest = []
            for i, idx in enumerate(indices[0]):
                if 0 <= idx < len(RetrievalAgent._examples):
                    # FAISS Inner Product with normalized vectors = cosine similarity
                    similarity = float(distances[0][i])
                    nearest.append(
                        {
                            "text": RetrievalAgent._examples[idx]["text"],
                            "label": RetrievalAgent._examples[idx]["label"],
                            "similarity": similarity,
                        }
                    )

            return nearest

        except Exception:
            return []

    def _build_mafa_prompt(
        self, text: str, nearest: List[Dict], labels: List[str]
    ) -> str:
        """Build ARQ-style prompt with retrieval results.

        MAFA Section 4.2.1: ARQ prompts with retrieval-augmented reasoning.
        Output MUST be valid JSON (NO FALLBACK).

        NOTE: This method ensures FAISS model is loaded before building prompt.
        """
        # Ensure FAISS model is loaded
        self._ensure_loaded()

        arq_prompt = ARQPromptBuilder.build_toxicity_arq(
            text=text, examples=nearest, agent_type="retrieval"
        )
        return ARQPromptBuilder.to_prompt(arq_prompt)

    def _parse_response(self, content: str) -> Dict[str, Any]:
        """Parse ARQ-style response from LLM.

        NO FALLBACK - Response MUST be valid JSON.
        """
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
    ) -> RetrievalAnnotation:
        """Annotate text using MAFA pattern: retrieval + LLM classification."""
        if labels is None or len(labels) == 0:
            labels = ["0", "1"]

        # 1. Lazy load model
        self._ensure_loaded()

        if RetrievalAgent._faiss_index is None or self._llm_client is None:
            return RetrievalAnnotation(
                label="unknown",
                confidence=0.5,
                nearest_examples=[],
            )

        # 2. Retrieve similar examples
        nearest = self._retrieve(text, k=3)  # Reduced to 3 for lower token usage

        if not nearest:
            return RetrievalAnnotation(
                label="unknown",
                confidence=0.3,
                nearest_examples=[],
            )

        # 3. Build MAFA-style prompt
        prompt = self._build_mafa_prompt(text, nearest, labels)

        # 4. LLM makes classification decision
        try:
            response = await self._llm_client.chat(
                [{"role": "user", "content": prompt}]
            )
            result = self._parse_response(response.content)
        except Exception as e:
            result = {
                "topic": "unknown",
                "confidence": 0.5,
                "confidence_level": "LOW",
                "reasoning": f"Error: {str(e)}",
                "intent_analysis": "",
            }

        # 5. Confidence is already numeric from ARQ parsing
        numeric_conf = result.get("confidence", 0.5)

        return RetrievalAnnotation(
            label=result.get("topic", "unknown"),
            confidence=numeric_conf,
            nearest_examples=nearest,
        )
