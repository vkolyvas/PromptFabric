"""Response Post-Processor for formatting, validation, and hallucination detection."""

import re
from typing import Any, Dict, Optional

from config.settings import settings
from llm_gateway import llm_gateway


class ResponsePostProcessor:
    """Post-processes LLM responses for quality assurance."""

    def __init__(self):
        self.llm = llm_gateway
        self.enabled = settings.enable_post_processor
        self.validator_model = settings.validator_model

    def process(
        self,
        response: str,
        original_prompt: str,
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Process and validate the LLM response."""
        if not self.enabled:
            return {
                "response": response,
                "valid": True,
                "issues": [],
                "validated": False,
            }

        issues = []

        # Step 1: Basic validation (always runs)
        basic_issues = self._basic_validation(response)
        issues.extend(basic_issues)

        # Step 2: Format the response
        formatted_response = self._format_response(response)

        # Step 3: Optional LLM-based validation (if validator model is different)
        if self.validator_model and self.validator_model != settings.generator_model:
            validation_result = self._validate_with_llm(
                formatted_response, original_prompt, context
            )
            issues.extend(validation_result.get("issues", []))
            validation_passed = validation_result.get("passed", True)
        else:
            validation_passed = len(issues) == 0

        return {
            "response": formatted_response,
            "valid": validation_passed,
            "issues": issues,
            "validated": True,
        }

    def _basic_validation(self, response: str) -> list:
        """Perform basic rule-based validation."""
        issues = []

        if not response or len(response.strip()) == 0:
            issues.append("empty_response")

        # Check for common hallucination markers
        hallucination_patterns = [
            r"^I don't know because",
            r"^As of my knowledge cutoff",
            r"^This information might be outdated",
        ]

        for pattern in hallucination_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                issues.append("potential_hallucination")

        # Check response length (too short might be incomplete)
        if len(response) < 10:
            issues.append("response_too_short")

        return issues

    def _format_response(self, response: str) -> str:
        """Format the response for better readability."""
        # Remove excessive whitespace
        response = re.sub(r"\n{3,}", "\n\n", response)
        response = response.strip()

        # Ensure proper code block formatting
        if "```" in response:
            # Fix code blocks without language
            response = re.sub(r"```\n", "```\n", response)

        return response

    def _validate_with_llm(
        self, response: str, original_prompt: str, context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Use a smaller LLM to validate the response."""
        validation_prompt = f"""You are a response validator. Your task is to check if the response is accurate and relevant to the user's prompt.

User's original prompt: {original_prompt}

Context provided:
{context if context else "No additional context"}

Response to validate:
{response}

Evaluate the response and respond with ONLY one word:
- "VALID" if the response is accurate, relevant, and helpful
- "INVALID" if the response is inaccurate, irrelevant, or contains hallucinations

Response:"""

        try:
            result = self.llm.generate(
                prompt=validation_prompt,
                model=self.validator_model,
                temperature=0.1,
                max_tokens=10,
            )

            result = result.strip().upper()
            passed = "VALID" in result

            if not passed:
                issues = ["llm_validation_failed"]
            else:
                issues = []

            return {"passed": passed, "issues": issues}

        except Exception as e:
            # If validation fails, assume response is valid to avoid blocking
            return {"passed": True, "issues": [f"validation_error: {str(e)}"]}


# Singleton instance
post_processor = ResponsePostProcessor()
