"""Query Expander - Embedding-based query expansion (MAFA-inspired).

Uses semantic similarity from training data to expand queries.
Inspired by MAFA Query Planning Agent but adapted for:
- No extra LLM calls (uses embeddings instead)
- Rule-free expansion (learns from training data)
- Fast inference (pre-computed embeddings)

Flow:
  Input: "tôi thấy người lái xe hơi bấm còi"
    ↓
  QueryExpander: Find similar terms from training data
    ↓
  Output: "tôi thấy người lái xe hơi bấm còi lái xe giao thông vi phạm"
"""

from typing import List, Dict, Optional, Set
import threading
import os


class QueryExpander:
    """Expand queries using semantic similarity from training data.

    Inspired by MAFA Query Planning Agent but adapted to avoid extra LLM calls.

    Usage:
        expander = QueryExpander("data/train.csv")
        expanded = expander.expand("comment text here")
    """

    _lock = threading.Lock()
    _model_loaded = False
    _embedding_model = None
    _vocabulary_embeddings = None
    _vocabulary_terms: List[str] = []

    def __init__(self, training_data_path: str, max_vocab_size: int = 10000):
        """Initialize query expander with training data."""
        self.training_data_path = training_data_path
        self.max_vocab_size = max_vocab_size
        self._ensure_loaded()

    def _ensure_loaded(self):
        """Ensure model and vocabulary embeddings are loaded."""
        if (
            QueryExpander._model_loaded
            and QueryExpander._vocabulary_embeddings is not None
        ):
            return

        with QueryExpander._lock:
            if (
                QueryExpander._model_loaded
                and QueryExpander._vocabulary_embeddings is not None
            ):
                return

            try:
                from sentence_transformers import SentenceTransformer
                import numpy as np
                import csv

                # Load embedding model
                QueryExpander._embedding_model = SentenceTransformer(
                    "sentence-transformers/all-MiniLM-L6-v2"
                )

                # Load vocabulary from training data
                terms = self._load_vocabulary()
                QueryExpander._vocabulary_terms = terms[: self.max_vocab_size]

                # Pre-compute embeddings for all terms (one-time cost)
                QueryExpander._vocabulary_embeddings = (
                    QueryExpander._embedding_model.encode(
                        QueryExpander._vocabulary_terms,
                        normalize_embeddings=True,
                        show_progress_bar=False,
                    )
                )

                QueryExpander._model_loaded = True
                print(
                    f"✓ QueryExpander loaded {len(QueryExpander._vocabulary_terms)} terms"
                )

            except ImportError as e:
                print(f"Warning: sentence-transformers not installed: {e}")
                QueryExpander._embedding_model = None
                QueryExpander._vocabulary_embeddings = None

    def _load_vocabulary(self) -> List[str]:
        """Extract unique meaningful terms from training data ONLY.

        NO hardcoding - learns everything from training data.
        """
        import csv

        terms: Set[str] = set()

        try:
            with open(self.training_data_path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)

                for row in reader:
                    # Extract text from various columns
                    for value in row.values():
                        if isinstance(value, str) and len(value) > 2:
                            # Split by common delimiters and clean
                            for word in value.split():
                                word = word.strip(".,!?;:()[]{}'\"")
                                if len(word) >= 3:
                                    terms.add(word.lower())

        except Exception as e:
            print(f"Warning: Failed to load training data: {e}")

        # Filter: remove short terms, numbers, and weird artifacts
        filtered_terms = []
        for term in terms:
            # Skip short terms
            if len(term) < 3:
                continue
            # Skip terms with numbers
            if any(c.isdigit() for c in term):
                continue
            # Skip terms with special characters (parsing artifacts)
            if any(c in term for c in ".0123456789"):
                continue
            # Skip non-Vietnamese terms
            if not any(
                c
                in "áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ"
                for c in term
            ):
                continue
            filtered_terms.append(term)

        return filtered_terms

    def _find_similar_terms(self, query: str, top_k: int = 5) -> List[str]:
        """Find semantically similar terms from vocabulary."""
        if (
            QueryExpander._embedding_model is None
            or QueryExpander._vocabulary_embeddings is None
        ):
            return []

        try:
            import numpy as np

            # Encode query
            query_embedding = QueryExpander._embedding_model.encode(
                [query], normalize_embeddings=True
            )

            # Compute similarity with all vocabulary terms
            similarities = np.dot(
                QueryExpander._vocabulary_embeddings, query_embedding.T
            ).flatten()

            # Get top-k most similar terms (excluding terms already in query)
            query_words = set(query.lower().split())
            similar_terms = []

            for idx in np.argsort(similarities)[::-1]:
                term = QueryExpander._vocabulary_terms[idx]
                # Skip if term is already in query
                if term in query_words:
                    continue
                # Skip if similarity is too low
                if similarities[idx] < 0.3:
                    break
                similar_terms.append(term)
                if len(similar_terms) >= top_k:
                    break

            return similar_terms

        except Exception:
            return []

    def expand(self, query: str, top_k: int = 5) -> str:
        """Expand query with semantically similar terms.

        Args:
            query: Original query text
            top_k: Number of similar terms to add (default: 5)

        Returns:
            Expanded query with similar terms appended
        """
        # Find similar terms
        similar_terms = self._find_similar_terms(query, top_k)

        if not similar_terms:
            return query

        # Append similar terms to query
        expanded = f"{query} {' '.join(similar_terms)}"

        return expanded

    def get_expansion_info(self, query: str, top_k: int = 5) -> Dict:
        """Get detailed info about query expansion."""
        similar_terms = self._find_similar_terms(query, top_k)

        return {
            "original_query": query,
            "expanded_query": self.expand(query, top_k),
            "added_terms": similar_terms,
            "num_terms_added": len(similar_terms),
            "vocabulary_size": len(QueryExpander._vocabulary_terms),
        }

    def stats(self) -> Dict:
        """Get expander statistics."""
        return {
            "vocabulary_size": len(QueryExpander._vocabulary_terms),
            "model_loaded": QueryExpander._model_loaded,
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        }
