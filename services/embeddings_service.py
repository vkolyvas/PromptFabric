"""Embeddings service using sentence-transformers."""

from typing import List, Optional

from config.settings import settings


class EmbeddingsService:
    """Generate embeddings for text using sentence-transformers."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or "sentence-transformers/all-MiniLM-L6-v2"
        self._model = None

    def _get_model(self):
        """Lazy load the model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError("sentence-transformers is required for embeddings")
        return self._model

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        model = self._get_model()
        embedding = model.encode(text, show_progress_bar=False)
        return embedding.tolist()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        model = self._get_model()
        embeddings = model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()


# Singleton instance
embeddings_service = EmbeddingsService()
