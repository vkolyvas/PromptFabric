import re
from typing import Any, Dict, Optional


class ResponseFormatter:
    """Post-processes and formats LLM responses"""

    @staticmethod
    def format(response: str, format_type: str = "markdown") -> str:
        """Format response based on type"""
        if format_type == "markdown":
            return ResponseFormatter._format_markdown(response)
        elif format_type == "plain":
            return ResponseFormatter._format_plain(response)
        elif format_type == "json":
            return ResponseFormatter._format_json(response)
        return response

    @staticmethod
    def _format_markdown(text: str) -> str:
        """Ensure proper markdown formatting"""
        # Fix common markdown issues
        text = text.strip()
        return text

    @staticmethod
    def _format_plain(text: str) -> str:
        """Strip markdown formatting"""
        # Remove markdown syntax
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
        text = re.sub(r"\*(.+?)\*", r"\1", text)
        text = re.sub(r"`(.+?)`", r"\1", text)
        text = re.sub(r"```[\s\S]*?```", "", text)
        return text.strip()

    @staticmethod
    def _format_json(text: str) -> str:
        """Extract and format JSON from response"""
        # Try to extract JSON from markdown code blocks
        match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
        if match:
            return match.group(1)

        # Try to find raw JSON
        match = re.search(r"(\{[\s\S]*?\})", text)
        if match:
            return match.group(1)

        return text

    @staticmethod
    def validate(response: str, min_length: int = 1) -> bool:
        """Validate response quality"""
        if not response or len(response.strip()) < min_length:
            return False
        return True

    @staticmethod
    def remove_hallucinations(response: str, known_facts: list = None) -> str:
        """Attempt to remove potential hallucinations"""
        # Basic placeholder - full implementation would use fact-checking
        return response


response_formatter = ResponseFormatter()
