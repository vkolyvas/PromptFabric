from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from config.hardware_detect import (
    detect_hardware,
    get_recommended_models,
    get_recommended_provider,
)
from config.settings import settings
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


@app.post("/memory")
async def create_session():
    """Create a new session"""
    try:
        new_session_id = memory_service.create_session()
        return {"session_id": new_session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memory/{session_id}")
async def create_session_with_id(session_id: Optional[str] = None):
    """Create a new session with optional custom ID"""
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


# Hardware & LLM Management Endpoints
@app.get("/hardware/detect")
async def hardware_detect():
    """Detect hardware and return recommendations"""
    hardware = detect_hardware()
    provider = get_recommended_provider(hardware)
    models = get_recommended_models(provider, hardware)

    return {
        "os_type": hardware.os_type,
        "total_ram_gb": hardware.total_ram_gb,
        "cpu_cores": hardware.cpu_cores,
        "has_nvidia_gpu": hardware.has_nvidia_gpu,
        "has_apple_silicon": hardware.has_apple_silicon,
        "has_amd_gpu": hardware.has_amd_gpu,
        "recommended_provider": provider,
        "recommended_models": models,
        "post_processor_enabled": settings.enable_post_processor,
    }


@app.get("/settings")
async def get_settings():
    """Get current settings"""
    return {
        "llm_provider": settings.llm_provider,
        "generator_model": settings.generator_model,
        "refiner_model": settings.refiner_model,
        "validator_model": settings.validator_model,
        "enable_post_processor": settings.enable_post_processor,
    }


@app.post("/settings")
async def update_settings(request: dict):
    """Update settings"""
    if "enable_post_processor" in request:
        settings.enable_post_processor = request["enable_post_processor"]
    if "llm_provider" in request:
        settings.llm_provider = request["llm_provider"]
    if "generator_model" in request:
        settings.generator_model = request["generator_model"]
    if "refiner_model" in request:
        settings.refiner_model = request["refiner_model"]

    return {"status": "updated", "settings": await get_settings()}


@app.post("/llm/start-ollama")
async def start_ollama():
    """Start Ollama service"""
    import os
    import subprocess

    try:
        # Check if already running
        import requests

        try:
            r = requests.get("http://localhost:11434/api/tags", timeout=2)
            if r.status_code == 200:
                return {"success": True, "message": "Ollama already running"}
        except:
            pass

        # Try to start ollama serve
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return {"success": True, "message": "Ollama started"}
    except Exception as e:
        return {"success": False, "message": str(e)}


@app.post("/llm/start-lmstudio")
async def start_lmstudio():
    """Open LM Studio application"""
    import platform
    import subprocess

    try:
        system = platform.system()
        if system == "Darwin":
            subprocess.run(["open", "-a", "LM Studio"], check=True)
        elif system == "Linux":
            subprocess.run(["lm-studio"], check=False)
        elif system == "Windows":
            subprocess.run(["start", "lm-studio"], shell=True, check=False)
        return {"success": True, "message": "LM Studio opened"}
    except Exception as e:
        return {"success": False, "message": f"Open LM Studio manually: {str(e)}"}


@app.post("/llm/pull-models")
async def pull_models(request: dict):
    """Pull models for the selected provider"""
    import subprocess

    import requests

    provider = request.get("provider", "ollama")
    generator = request.get("generator_model", "llama3.2:3b")
    refiner = request.get("refiner_model", "gemma:2b")

    if provider != "ollama":
        return {"success": False, "message": "For LM Studio, load models in the app"}

    try:
        # Check if ollama is running
        try:
            r = requests.get("http://localhost:11434/api/tags", timeout=2)
            if r.status_code != 200:
                return {"success": False, "message": "Start Ollama first"}
        except:
            return {"success": False, "message": "Start Ollama first"}

        # Pull models
        for model in [generator, refiner]:
            subprocess.run(
                ["ollama", "pull", model],
                capture_output=True,
                timeout=300,
            )

        return {"success": True, "message": f"Pulled {generator} and {refiner}"}
    except Exception as e:
        return {"success": False, "message": str(e)}


@app.get("/llm/status")
async def llm_status():
    """Check LLM provider status"""
    import requests

    status = {
        "ollama_running": False,
        "lm_studio_running": False,
    }

    # Check Ollama
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        if r.status_code == 200:
            status["ollama_running"] = True
    except:
        pass

    # Check LM Studio
    try:
        r = requests.get("http://localhost:1234/v1/models", timeout=2)
        if r.status_code == 200:
            status["lm_studio_running"] = True
    except:
        pass

    return status


if __name__ == "__main__":
    import uvicorn

    from config.settings import settings

    uvicorn.run(app, host=settings.api_host, port=settings.api_port)
