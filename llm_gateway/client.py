import requests
from typing import List, Dict, Any, Optional
from config.settings import settings


class LLMGateway:
    """Gateway for communicating with local LLM via LM Studio"""

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or settings.lm_studio_url

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Send chat completion request to LM Studio"""

        # Build messages with system prompt if provided
        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        payload = {
            "model": model or "gemma-3-4b-it",
            "messages": full_messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            result = response.json()

            return {
                "content": result["choices"][0]["message"]["content"],
                "model": result.get("model", model),
                "usage": result.get("usage", {}),
                "finish_reason": result["choices"][0].get("finish_reason")
            }
        except requests.exceptions.RequestException as e:
            raise Exception(f"LLM request failed: {str(e)}")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> str:
        """Simple generation method"""
        messages = [{"role": "user", "content": prompt}]
        result = self.chat_completion(
            messages=messages,
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return result["content"]


# Singleton instance
llm_gateway = LLMGateway()
