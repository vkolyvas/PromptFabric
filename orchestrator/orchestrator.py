from typing import Dict, Any, Optional
from llm_gateway import llm_gateway
from orchestrator.prompt_refiner import prompt_refiner
from orchestrator.context_builder import context_builder
from orchestrator.memory_manager import memory_manager
from config.settings import settings


class PromptOrchestrator:
    """Main orchestration brain that coordinates all components"""

    def __init__(self):
        self.llm = llm_gateway
        self.refiner = prompt_refiner
        self.context = context_builder
        self.memory = memory_manager

    def process(
        self,
        message: str,
        session_id: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> Dict[str, Any]:
        """Main orchestration pipeline"""

        # Create or get session
        if not session_id:
            session_id = self.memory.create_session()
        else:
            # Ensure session exists
            self.memory.create_session(session_id)

        # Get conversation history
        history = self.memory.get_session_history(session_id)

        # Search for relevant context
        context_results = self.context.search(message, top_k=5)
        context_text = "\n".join([r.get("content", "") for r in context_results])

        # Refine the user's prompt
        refined_prompt = self.refiner.refine(message, context_text)

        # Build messages for the generator
        messages = history.copy()
        messages.append({"role": "user", "content": refined_prompt})

        # Generate response
        try:
            response = self.llm.chat_completion(
                messages=messages,
                model=model or settings.generator_model,
                system_prompt=settings.generator_system_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )

            content = response["content"]

            # Store in memory
            self.memory.add_message(session_id, "user", message)
            self.memory.add_message(session_id, "assistant", content)

            return {
                "response": content,
                "session_id": session_id,
                "model": response.get("model", settings.generator_model),
                "refined_prompt": refined_prompt,
                "context_used": len(context_results) > 0
            }

        except Exception as e:
            return {
                "response": f"Error processing request: {str(e)}",
                "session_id": session_id,
                "model": model or settings.generator_model,
                "error": True
            }

    def refine_prompt(self, prompt: str, context: Optional[str] = None) -> str:
        """Standalone prompt refinement"""
        return self.refiner.refine(prompt, context)

    def search_context(self, query: str, top_k: int = 5):
        """Standalone context search"""
        return self.context.search(query, top_k)


orchestrator = PromptOrchestrator()
