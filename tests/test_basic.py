import pytest

from models.schemas import ChatRequest, PromptRefineRequest


def test_chat_request_schema():
    """Test ChatRequest model"""
    request = ChatRequest(message="Hello, world!")
    assert request.message == "Hello, world!"
    assert request.session_id is None
    assert request.temperature == 0.7


def test_prompt_refine_request():
    """Test PromptRefineRequest model"""
    request = PromptRefineRequest(prompt="Test prompt")
    assert request.prompt == "Test prompt"
    assert request.context is None


def test_prompt_refine_with_context():
    """Test PromptRefineRequest with context"""
    request = PromptRefineRequest(prompt="Test", context="Some context")
    assert request.context == "Some context"
