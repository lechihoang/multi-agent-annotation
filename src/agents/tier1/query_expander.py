

from typing import List, Dict, Optional, Set
import threading
import os


class QueryExpander:

    _lock = threading.Lock()
    _model_loaded = False
    _embedding_model = None
    _vocabulary_embeddings = None
    _vocabulary_terms: List[str] = []

    def __init__(self, training_data_path: str, max_vocab_size: int = 10000):
        self.training_data_path = training_data_path
        self.max_vocab_size = max_vocab_size
        self._ensure_loaded()

    def _ensure_loaded(self):
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

                QueryExpander._embedding_model = SentenceTransformer(
                    "sentence-transformers/all-MiniLM-L6-v2"
                )

                terms = self._load_vocabulary()
                QueryExpander._vocabulary_terms = terms[: self.max_vocab_size]

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
        import csv

        terms: Set[str] = set()

        try:
            with open(self.training_data_path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)

                for row in reader:
                    for value in row.values():
                        if isinstance(value, str) and len(value) > 2:
                            for word in value.split():
                                word = word.strip(".,!?;:()[]{}'\"")
                                if len(word) >= 3:
                                    terms.add(word.lower())

        except Exception as e:
            print(f"Warning: Failed to load training data: {e}")

        filtered_terms = []
        for term in terms:
            if len(term) < 3:
                continue
            if any(c.isdigit() for c in term):
                continue
            if any(c in term for c in ".0123456789"):
                continue
            if not any(
                c
                in "áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ"
                for c in term
            ):
                continue
            filtered_terms.append(term)

        return filtered_terms

    def _find_similar_terms(self, query: str, top_k: int = 5) -> List[str]:
        if (
            QueryExpander._embedding_model is None
            or QueryExpander._vocabulary_embeddings is None
        ):
            return []

        import numpy as np

        query_embedding = QueryExpander._embedding_model.encode(
            [query], normalize_embeddings=True
        )

        similarities = np.dot(
            QueryExpander._vocabulary_embeddings, query_embedding.T
        ).flatten()

        query_words = set(query.lower().split())
        similar_terms = []

        for idx in np.argsort(similarities)[::-1]:
            term = QueryExpander._vocabulary_terms[idx]
            if term in query_words:
                continue
            if similarities[idx] < 0.3:
                break
            similar_terms.append(term)
            if len(similar_terms) >= top_k:
                break

        return similar_terms

    def expand(self, query: str, top_k: int = 5) -> str:
        similar_terms = self._find_similar_terms(query, top_k)

        if not similar_terms:
            return query

        expanded = f"{query} {' '.join(similar_terms)}"

        return expanded

    def get_expansion_info(self, query: str, top_k: int = 5) -> Dict:
        similar_terms = self._find_similar_terms(query, top_k)

        return {
            "original_query": query,
            "expanded_query": self.expand(query, top_k),
            "added_terms": similar_terms,
            "num_terms_added": len(similar_terms),
            "vocabulary_size": len(QueryExpander._vocabulary_terms),
        }

    def stats(self) -> Dict:
        return {
            "vocabulary_size": len(QueryExpander._vocabulary_terms),
            "model_loaded": QueryExpander._model_loaded,
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        }
