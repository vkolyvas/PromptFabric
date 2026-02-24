from typing import Any, Dict, List, Optional

from config.settings import settings


class ContextBuilder:
    """RAG context injection using vector search"""

    def __init__(self):
        self.vector_db_type = settings.vector_db_type
        self._client = None
        self._collection = None

    def _init_chroma(self):
        """Initialize ChromaDB client"""
        try:
            import chromadb

            if not self._client:
                self._client = chromadb.PersistentClient(
                    path=settings.chroma_persist_directory
                )
            return self._client
        except ImportError:
            return None

    def _init_qdrant(self):
        """Initialize Qdrant client"""
        try:
            from qdrant_client import QdrantClient

            if not self._client:
                self._client = QdrantClient(url=settings.qdrant_url)
            return self._client
        except ImportError:
            return None

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant context"""

        if self.vector_db_type == "chroma":
            return self._search_chroma(query, top_k)
        elif self.vector_db_type == "qdrant":
            return self._search_qdrant(query, top_k)
        return []

    def _search_chroma(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Search using ChromaDB"""
        try:
            client = self._init_chroma()
            if not client:
                return []

            # Get or create collection
            collection = client.get_or_create_collection("context")
            results = collection.query(query_texts=[query], n_results=top_k)

            formatted_results = []
            if results["documents"]:
                for i, doc in enumerate(results["documents"][0]):
                    formatted_results.append(
                        {
                            "content": doc,
                            "id": results["ids"][0][i] if results.get("ids") else None,
                            "distance": (
                                results["distances"][0][i]
                                if results.get("distances")
                                else None
                            ),
                        }
                    )

            return formatted_results
        except Exception:
            return []

    def _search_qdrant(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Search using Qdrant"""
        try:
            client = self._init_qdrant()
            if not client:
                return []

            # Note: Full implementation would require embeddings
            # This is a simplified placeholder
            return []
        except Exception:
            return []

    def add_context(self, content: str, metadata: Optional[Dict] = None):
        """Add context to vector database"""
        if self.vector_db_type == "chroma":
            self._add_chroma(content, metadata)
        elif self.vector_db_type == "qdrant":
            self._add_qdrant(content, metadata)

    def _add_chroma(self, content: str, metadata: Optional[Dict]):
        """Add to ChromaDB"""
        try:
            client = self._init_chroma()
            if not client:
                return

            collection = client.get_or_create_collection("context")
            import uuid

            collection.add(
                documents=[content], ids=[str(uuid.uuid4())], metadatas=[metadata or {}]
            )
        except Exception:
            pass

    def _add_qdrant(self, content: str, metadata: Optional[Dict]):
        """Add to Qdrant"""
        # Placeholder for Qdrant implementation
        pass


context_builder = ContextBuilder()
