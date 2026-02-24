from orchestrator import prompt_refiner
from models.schemas import PromptRefineRequest, PromptRefineResponse


class PromptService:
    """Service for prompt refinement operations"""

    def refine_prompt(self, request: PromptRefineRequest) -> PromptRefineResponse:
        """Refine user prompt"""
        refined = prompt_refiner.refine(request.prompt, request.context)

        return PromptRefineResponse(
            refined_prompt=refined,
            original_prompt=request.prompt
        )


prompt_service = PromptService()
