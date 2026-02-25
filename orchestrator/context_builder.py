"""RAG context injection using vector search with embeddings."""

import os
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
                # Use persistence client for local storage
                chroma_path = settings.chroma_persist_directory
                os.makedirs(chroma_path, exist_ok=True)
                self._client = chromadb.PersistentClient(path=chroma_path)
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
        """Search using ChromaDB with embeddings"""
        try:
            client = self._init_chroma()
            if not client:
                return []

            # Get or create collection with embeddings
            try:
                collection = client.get_collection("context")
            except Exception:
                # Collection doesn't exist yet
                return []

            # Check if collection has embeddings
            if collection.count() == 0:
                return []

            # Query using embeddings
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
                            "metadata": (
                                results["metadatas"][0][i]
                                if results.get("metadatas")
                                else {}
                            ),
                        }
                    )

            return formatted_results
        except Exception:
            return []

    def _search_qdrant(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Search using Qdrant"""
        # Placeholder for Qdrant implementation
        return []

    def add_context(self, content: str, metadata: Optional[Dict] = None):
        """Add context to vector database"""
        if self.vector_db_type == "chroma":
            self._add_chroma(content, metadata)
        elif self.vector_db_type == "qdrant":
            self._add_qdrant(content, metadata)

    def _add_chroma(self, content: str, metadata: Optional[Dict]):
        """Add text directly to ChromaDB (legacy method)"""
        try:
            client = self._init_chroma()
            if not client:
                return

            import uuid

            collection = client.get_or_create_collection(
                "context", metadata={"hnsw:space": "cosine"}
            )

            # Check if we should use embeddings
            if self._has_embeddings():
                # Use embedding-based add
                self._add_with_embeddings(collection, [content], [metadata or {}])
            else:
                # Legacy: add raw text (Chroma will auto-embed)
                collection.add(
                    documents=[content],
                    ids=[str(uuid.uuid4())],
                    metadatas=[metadata or {}],
                )
        except Exception:
            pass

    def add_chunks(self, chunks: List[Dict[str, Any]], use_embeddings: bool = True):
        """Add multiple text chunks to vector database"""
        if self.vector_db_type == "chroma":
            self._add_chunks_chroma(chunks, use_embeddings)
        elif self.vector_db_type == "qdrant":
            self._add_chunks_qdrant(chunks)

    def _add_chunks_chroma(
        self, chunks: List[Dict[str, Any]], use_embeddings: bool = True
    ):
        """Add chunks to ChromaDB"""
        try:
            client = self._init_chroma()
            if not client:
                return

            collection = client.get_or_create_collection(
                "context", metadata={"hnsw:space": "cosine"}
            )

            documents = [chunk["content"] for chunk in chunks]
            metadatas = [chunk.get("metadata", {}) for chunk in chunks]
            ids = [f"chunk_{i}" for i in range(len(chunks))]

            if use_embeddings and self._has_embeddings():
                self._add_with_embeddings(collection, documents, metadatas, ids)
            else:
                # Let Chroma handle embeddings
                collection.add(
                    documents=documents,
                    ids=ids,
                    metadatas=metadatas,
                )
        except Exception:
            pass

    def _add_with_embeddings(
        self,
        collection,
        documents: List[str],
        metadatas: List[Dict],
        ids: Optional[List[str]] = None,
    ):
        """Add documents with pre-computed embeddings"""
        try:
            from services.embeddings_service import embeddings_service

            # Generate embeddings
            embeddings = embeddings_service.embed_texts(documents)

            # Add with embeddings
            if ids is None:
                import uuid

                ids = [str(uuid.uuid4()) for _ in documents]

            collection.add(
                embeddings=embeddings,
                documents=documents,
                ids=ids,
                metadatas=metadatas,
            )
        except Exception:
            # Fallback: let Chroma handle it
            import uuid

            if ids is None:
                ids = [str(uuid.uuid4()) for _ in documents]
            collection.add(
                documents=documents,
                ids=ids,
                metadatas=metadatas,
            )

    def _has_embeddings(self) -> bool:
        """Check if sentence-transformers is available"""
        try:
            from sentence_transformers import SentenceTransformer

            return True
        except ImportError:
            return False

    def _add_qdrant(self, content: str, metadata: Optional[Dict]):
        """Add to Qdrant"""
        # Placeholder for Qdrant implementation
        pass

    def _add_chunks_qdrant(self, chunks: List[Dict[str, Any]]):
        """Add chunks to Qdrant"""
        # Placeholder for Qdrant implementation
        pass

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        if self.vector_db_type == "chroma":
            return self._get_chroma_stats()
        return {"error": "Unsupported vector DB type"}

    def _get_chroma_stats(self) -> Dict[str, Any]:
        """Get ChromaDB stats"""
        try:
            client = self._init_chroma()
            if not client:
                return {"error": "Could not initialize ChromaDB"}

            try:
                collection = client.get_collection("context")
                count = collection.count()
                return {
                    "type": "chroma",
                    "total_documents": count,
                    "persist_directory": settings.chroma_persist_directory,
                }
            except Exception:
                return {
                    "type": "chroma",
                    "total_documents": 0,
                    "persist_directory": settings.chroma_persist_directory,
                }
        except Exception as e:
            return {"error": str(e)}


context_builder = ContextBuilder()
