from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import requests

from config.settings import settings


class BaseLLMGateway(ABC):
    """Abstract base class for LLM gateways"""

    @abstractmethod
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send chat completion request"""
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """Simple generation method"""
        pass


class LMStudioGateway(BaseLLMGateway):
    """Gateway for communicating with local LLM via LM Studio"""

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or settings.lm_studio_url

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send chat completion request to LM Studio"""

        # Build messages with system prompt if provided
        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        payload = {
            "model": model or settings.generator_model,
            "messages": full_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions", json=payload, timeout=120
            )
            response.raise_for_status()
            result = response.json()

            return {
                "content": result["choices"][0]["message"]["content"],
                "model": result.get("model", model),
                "usage": result.get("usage", {}),
                "finish_reason": result["choices"][0].get("finish_reason"),
            }
        except requests.exceptions.RequestException as e:
            raise Exception(f"LLM request failed: {str(e)}")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """Simple generation method"""
        messages = [{"role": "user", "content": prompt}]
        result = self.chat_completion(
            messages=messages,
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return result["content"]


class OllamaGateway(BaseLLMGateway):
    """Gateway for communicating with local LLM via Ollama"""

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or settings.ollama_url
        self.default_model = settings.ollama_model

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send chat completion request to Ollama"""

        # Build messages with system prompt if provided
        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        payload = {
            "model": model or self.default_model,
            "messages": full_messages,
            "temperature": temperature,
            "options": {
                "num_predict": max_tokens,
            },
            "stream": False,
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/chat", json=payload, timeout=120
            )
            response.raise_for_status()
            result = response.json()

            return {
                "content": result["message"]["content"],
                "model": result.get("model", model),
                "usage": {
                    "prompt_tokens": result.get("prompt_eval_count", 0),
                    "completion_tokens": result.get("eval_count", 0),
                    "total_tokens": result.get("prompt_eval_count", 0)
                    + result.get("eval_count", 0),
                },
                "finish_reason": result.get("done_reason", "stop"),
            }
        except requests.exceptions.RequestException as e:
            raise Exception(f"Ollama request failed: {str(e)}")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """Simple generation method using /api/generate endpoint"""
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"System: {system_prompt}\n\nUser: {prompt}"

        payload = {
            "model": model or self.default_model,
            "prompt": full_prompt,
            "temperature": temperature,
            "options": {
                "num_predict": max_tokens,
            },
            "stream": False,
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/generate", json=payload, timeout=120
            )
            response.raise_for_status()
            result = response.json()

            return result.get("response", "")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Ollama generate request failed: {str(e)}")

    def list_models(self) -> List[Dict[str, Any]]:
        """List available models in Ollama"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=30)
            response.raise_for_status()
            result = response.json()
            return result.get("models", [])
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to list Ollama models: {str(e)}")


# Factory function to get the appropriate gateway
def get_llm_gateway(
    provider: Optional[str] = None,
) -> BaseLLMGateway:
    """Get the appropriate LLM gateway based on provider setting"""
    provider = provider or settings.llm_provider

    if provider == "ollama":
        return OllamaGateway()
    else:
        return LMStudioGateway()


# Backwards compatibility - keep existing class name as alias
LLMGateway = LMStudioGateway

# Singleton instances
llm_gateway = get_llm_gateway()
