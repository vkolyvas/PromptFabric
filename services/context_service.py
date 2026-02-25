"""Service for context/RAG operations."""

from typing import Any, Dict, List

from models.schemas import (
    AddContextRequest,
    ContextSearchRequest,
    ContextSearchResponse,
)
from orchestrator import context_builder
from services.document_processor import document_processor


class ContextService:
    """Service for context/RAG operations"""

    def search(self, request: ContextSearchRequest) -> ContextSearchResponse:
        """Search for relevant context"""
        results = context_builder.search(request.query, request.top_k or 5)

        return ContextSearchResponse(results=results, query=request.query)

    def add_context(self, content: str, metadata: dict = None):
        """Add context to vector store"""
        context_builder.add_context(content, metadata)

    def add_file(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """Process and add a file to the vector store"""
        # Process the file into chunks
        chunks = document_processor.process_file(file_content, filename)

        # Add chunks to vector database
        context_builder.add_chunks(chunks, use_embeddings=True)

        return {
            "filename": filename,
            "chunks_added": len(chunks),
            "status": "success",
        }

    def add_text_chunks(self, chunks: List[Dict[str, Any]]):
        """Add pre-processed text chunks to vector store"""
        context_builder.add_chunks(chunks, use_embeddings=True)

    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        return context_builder.get_stats()


context_service = ContextService()
