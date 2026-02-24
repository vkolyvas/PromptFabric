from models.schemas import MemoryResponse
from orchestrator import memory_manager


class MemoryService:
    """Service for memory/conversation history operations"""

    def get_memory(self, session_id: str) -> MemoryResponse:
        """Get conversation memory"""
        messages = memory_manager.get_messages(session_id)

        return MemoryResponse(
            session_id=session_id, messages=messages, total_count=len(messages)
        )

    def create_session(self, session_id: str = None) -> str:
        """Create a new session"""
        return memory_manager.create_session(session_id)

    def delete_session(self, session_id: str):
        """Delete a session"""
        memory_manager.delete_session(session_id)


memory_service = MemoryService()
