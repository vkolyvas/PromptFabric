# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PromptFabric is a local LLM orchestration system that replicates cloud-level AI behavior using local models. It places an intelligent orchestration layer between users and local LLMs (via LM Studio) to achieve production-quality responses.

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

### The Orchestrator

The Prompt Orchestrator is the "brain" that:
- Rewrites prompts
- Selects appropriate model
- Injects context and system prompts
- Validates output

## Commands

### Setup
```bash
# Install dependencies
pip install fastapi uvicorn chromadb qdrant sqlalchemy redis

# Start LM Studio with models loaded
# Ensure LM Studio API is running at http://localhost:1234/v1/chat/completions
```

### Running the Application
```bash
# Run API Gateway
uvicorn api_gateway.main:app --host 0.0.0.0 --port 8000

# Or run all services via Docker
docker-compose up
```

### Testing
```bash
# Run specific test
pytest tests/test_orchestrator.py -v

# Run all tests
pytest
```

## Model Configuration

Recommended model assignments:
- **Prompt Refiner**: Gemma 3 / TinyLlama (small, fast)
- **Main Generator**: DeepSeek-Coder R1 7B
- **Validator**: Phi-3 Mini (small validation model)

## LM Studio Integration

Connect to local models via:
```
http://localhost:1234/v1/chat/completions
```

## Data Storage

- **Vector DB**: ChromaDB or Qdrant (for RAG/context)
- **Memory**: SQLite (for conversation history)
- **Cache**: Redis (optional, for embeddings/responses)

## Deployment Options

1. **GPU Mode**: Full performance with GPU acceleration (current)
2. **CPU Only Mode**: Reduced performance for CPU-only environments

## API Endpoints (FastAPI)

- `POST /chat` - Main chat endpoint
- `POST /prompt/refine` - Prompt refinement
- `GET /context/search` - Context search
- `GET /memory/{session_id}` - Retrieve conversation memory
