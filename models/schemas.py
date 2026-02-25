from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class ChatRequest(BaseModel):
    """Request model for chat endpoint"""

    message: str
    session_id: Optional[str] = None
    model: Optional[str] = None
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 2048


class ChatResponse(BaseModel):
    """Response model for chat endpoint"""

    response: str
    session_id: str
    model: str
    tokens_used: Optional[int] = None


class PromptRefineRequest(BaseModel):
    """Request model for prompt refinement"""

    prompt: str
    context: Optional[str] = None


class PromptRefineResponse(BaseModel):
    """Response model for prompt refinement"""

    refined_prompt: str
    original_prompt: str


class ContextSearchRequest(BaseModel):
    """Request model for context search"""

    query: str
    top_k: Optional[int] = 5


class ContextSearchResponse(BaseModel):
    """Response model for context search"""

    results: List[Dict[str, Any]]
    query: str


class AddContextRequest(BaseModel):
    """Request model for adding context"""

    content: str
    metadata: Optional[Dict[str, Any]] = None


class MemoryResponse(BaseModel):
    """Response model for memory retrieval"""

    session_id: str
    messages: List[Dict[str, Any]]
    total_count: int
