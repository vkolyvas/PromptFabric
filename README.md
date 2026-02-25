# PromptFabric

A local LLM orchestration system that replicates cloud-level AI behavior using local models. It places an intelligent orchestration layer between users and local LLMs (via LM Studio) to achieve production-quality responses.

## Architecture

```
User → Prompt Refiner → Context Injector → Memory Injector → LLM Gateway → Response Post-Processor → User
```

### Core Components

1. **Prompt Pre-Processor (Prompt Refiner)**: Converts unstructured user input into optimized structured prompts using a small/fast model (e.g., Gemma, TinyLlama)
2. **System Prompt Layer**: Persistent hidden behavior control (user never sees this)
3. **Context Builder (RAG Layer)**: Injects documentation, logs, schemas via vector search (ChromaDB or Qdrant)
4. **Prompt Refiner Model**: Dual-model pattern - Refiner optimizes prompts, Generator produces answers
5. **Response Post-Processor**: Formats, validates, removes hallucinations
6. **Memory Layer**: Stores conversation history, user preferences in SQLite

### Prompt Flow (Step by Step)

```
USER INPUT: "How do I fix this bug?"
│
├─ STEP 1: MEMORY LAYER (SQLite)
│  └─ Load conversation history for session
│
├─ STEP 2: CONTEXT BUILDER (RAG - ChromaDB/Qdrant)
│  └─ Search vector DB for relevant docs/code (top K=5)
│
├─ STEP 3: PROMPT REFINER (Small Model: gemma:2b)
│  └─ Transform: "How do I fix this bug?"
│     → "Explain how to debug a Python application.
│        Include: 1) Using pdb/IPDB, 2) Print statements,
│        3) VS Code debugger. Provide examples."
│
├─ STEP 4: LLM GATEWAY (Main Model: llama3.2:3b)
│  └─ Build messages + Send to Ollama/LM Studio → Get response
│
├─ STEP 5: RESPONSE POST-PROCESSOR
│  └─ Format, validate, remove hallucinations
│
├─ STEP 6: MEMORY LAYER (Store)
│  └─ Save user message + assistant response to SQLite
│
└─ RESPONSE → User
```

| Step | Component | Model | Purpose |
|------|-----------|-------|---------|
| 1-2 | Memory/Context | N/A | Data retrieval |
| 3 | Prompt Refiner | gemma:2b | Optimize prompt structure |
| 4 | LLM Generator | llama3.2:3b | Generate final response |
| 5 | Post-Processor | N/A | Format/validate |

## Quick Start

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Start LM Studio with models loaded
# Ensure LM Studio API is running at http://localhost:1234/v1/chat/completions
```

### Running the Application

```bash
# Run API Gateway (port 8030)
uvicorn api_gateway.main:app --host 127.0.0.1 --port 8030

# Serve frontend (port 3030) - in separate terminal
cd frontend && python3 -m http.server 3030

# Or run all services via Docker
docker-compose up
```

The frontend at http://127.0.0.1:3030 provides:
- Auto hardware detection on load
- Provider selection (Ollama / LM Studio)
- One-click model pulling
- Chat interface with session management

## API Endpoints

### Chat & Memory
- `POST /chat` - Main chat endpoint
- `GET /memory/{session_id}` - Retrieve conversation memory
- `POST /memory/{session_id}` - Create new session
- `DELETE /memory/{session_id}` - Delete session

### Prompt & Context
- `POST /prompt/refine` - Prompt refinement
- `POST /context/search` - Context search

### Hardware & LLM Management
- `GET /hardware/detect` - Detect hardware and get recommendations
- `POST /llm/start-ollama` - Start Ollama service
- `POST /llm/start-lmstudio` - Open LM Studio application
- `POST /llm/pull-models` - Pull models for selected provider
- `GET /llm/status` - Check if Ollama/LM Studio is running

## Model Configuration

The system auto-detects your hardware and recommends the best provider and models.

### Hardware Recommendations

| Hardware | Recommended Provider | Generator Model | Refiner Model |
|----------|---------------------|-----------------|---------------|
| NVIDIA GPU | LM Studio | deepseek-coder-r1-7b | gemma-3-4b-it |
| Apple Silicon | Ollama | llama3.2:3b | gemma:2b |
| AMD GPU | Ollama | llama3.2:3b | gemma:2b |
| CPU Only (16GB+) | Ollama | llama3.2:3b | gemma:2b |
| CPU Only (<16GB) | Ollama | llama3.2:1b | gemma:1b |

### API Endpoints for LLM Management

- `GET /hardware/detect` - Detect hardware and get recommendations
- `POST /llm/start-ollama` - Start Ollama service
- `POST /llm/start-lmstudio` - Open LM Studio
- `POST /llm/pull-models` - Pull recommended models
- `GET /llm/status` - Check if Ollama/LM Studio is running

## LM Studio Integration

Connect to local models via:
```
http://localhost:1234/v1/chat/completions
```

## Data Storage

- **Vector DB**: ChromaDB or Qdrant (for RAG/context)
- **Memory**: SQLite (for conversation history)

## Development

```bash
# Run tests
pytest

# Run with custom settings
cp .env.example .env
# Edit .env with your configuration
```
