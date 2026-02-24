from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from models.schemas import (
    ChatRequest,
    ChatResponse,
    ContextSearchRequest,
    ContextSearchResponse,
    MemoryResponse,
    PromptRefineRequest,
    PromptRefineResponse,
)
from orchestrator import orchestrator
from services import context_service, memory_service, prompt_service

app = FastAPI(
    title="PromptFabric API",
    description="Local LLM Orchestration System",
    version="0.1.0",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "healthy", "service": "PromptFabric"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint"""
    try:
        result = orchestrator.process(
            message=request.message,
            session_id=request.session_id,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

        return ChatResponse(
            response=result["response"],
            session_id=result["session_id"],
            model=result["model"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/prompt/refine", response_model=PromptRefineResponse)
async def refine_prompt(request: PromptRefineRequest):
    """Refine user prompt"""
    try:
        result = prompt_service.refine_prompt(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/context/search", response_model=ContextSearchResponse)
async def search_context(request: ContextSearchRequest):
    """Search for relevant context"""
    try:
        result = context_service.search(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memory/{session_id}", response_model=MemoryResponse)
async def get_memory(session_id: str):
    """Get conversation memory"""
    try:
        result = memory_service.get_memory(session_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memory/{session_id}")
async def create_session(session_id: Optional[str] = None):
    """Create a new session"""
    try:
        new_session_id = memory_service.create_session(session_id)
        return {"session_id": new_session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/memory/{session_id}")
async def delete_session(session_id: str):
    """Delete a session"""
    try:
        memory_service.delete_session(session_id)
        return {"status": "deleted", "session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    from config.settings import settings

    uvicorn.run(app, host=settings.api_host, port=settings.api_port)
