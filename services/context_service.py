from orchestrator import context_builder
from models.schemas import ContextSearchRequest, ContextSearchResponse


class ContextService:
    """Service for context/RAG operations"""

    def search(self, request: ContextSearchRequest) -> ContextSearchResponse:
        """Search for relevant context"""
        results = context_builder.search(request.query, request.top_k or 5)

        return ContextSearchResponse(
            results=results,
            query=request.query
        )

    def add_context(self, content: str, metadata: dict = None):
        """Add context to vector store"""
        context_builder.add_context(content, metadata)


context_service = ContextService()
