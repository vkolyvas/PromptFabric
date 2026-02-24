from config.settings import settings
from llm_gateway import llm_gateway


class PromptRefiner:
    """Refines user prompts using a small/fast model"""

    def __init__(self):
        self.llm = llm_gateway
        self.system_prompt = settings.refiner_system_prompt

    def refine(self, prompt: str, context: str = None) -> str:
        """Convert unstructured prompt to optimized structured prompt"""

        refinement_prompt = f"""Refine the following user prompt to produce better LLM responses.

Original prompt: {prompt}
"""

        if context:
            refinement_prompt += f"\nRelevant context:\n{context}"

        refinement_prompt += """

Provide a refined, well-structured prompt that:
1. Clearly defines the task
2. Specifies desired format/style
3. Includes relevant constraints
4. Adds helpful context if needed

Refined prompt:"""

        try:
            response = self.llm.generate(
                prompt=refinement_prompt,
                system_prompt=self.system_prompt,
                model=settings.refiner_model,
                temperature=0.5,
                max_tokens=1024,
            )
            return response.strip()
        except Exception as e:
            # Fallback to original prompt if refinement fails
            return prompt


prompt_refiner = PromptRefiner()
