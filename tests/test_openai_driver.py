#!/usr/bin/env python3
"""Unit tests for openai_driver module"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from mi.external.openai_driver import sample
from mi.llm.data_models import SampleCfg, Chat, ChatMessage, MessageRole


@pytest.mark.asyncio
async def test_sample_with_logprobs_returns_non_empty_logprobs():
    """Test that when SampleCfg(logprobs=True) is used, a non-empty logprobs dict is returned"""
    
    # Create test data
    model_id = "gpt-4o-mini"
    input_chat = Chat(messages=[
        ChatMessage(role=MessageRole.user, content="Hello, world!")
    ])
    sample_cfg = SampleCfg(temperature=0.0, logprobs=True)
    
    # Mock the OpenAI API response with logprobs
    mock_choice = MagicMock()
    mock_choice.message.content = "Hello there!"
    mock_choice.finish_reason = "stop"
    
    # Create a mock logprobs object with content attribute
    mock_logprobs = MagicMock()
    mock_logprobs.content = [
        {"Hello": -0.1, "Hi": -0.5},
        {" there": -0.2, "!": -0.8},
        {"!": -0.3, ".": -1.2}
    ]
    mock_choice.logprobs = mock_logprobs
    
    mock_api_response = MagicMock()
    mock_api_response.choices = [mock_choice]
    
    # Mock the client and its chat.completions.create method
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_api_response)
    
    # Mock get_client_for_model to return our mock client
    with patch('mi.external.openai_driver.get_client_for_model', return_value=mock_client):
        # Call the function
        result = await sample(model_id, input_chat, sample_cfg)
        
        # Verify the result
        assert result.model_id == model_id
        assert result.completion == "Hello there!"
        assert result.stop_reason.value == "stop_sequence"
        assert result.logprobs is not None
        assert len(result.logprobs) == 3  # Should have 3 tokens
        
        # Verify the logprobs structure
        assert "Hello" in result.logprobs[0]
        assert result.logprobs[0]["Hello"] == -0.1
        assert "Hi" in result.logprobs[0]
        assert result.logprobs[0]["Hi"] == -0.5
        
        # Verify the API was called with logprobs=True
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args
        assert call_args[1]["logprobs"] is True


@pytest.mark.asyncio
async def test_sample_without_logprobs_returns_none():
    """Test that when SampleCfg(logprobs=False) is used, logprobs is None"""
    
    # Create test data
    model_id = "gpt-4o-mini"
    input_chat = Chat(messages=[
        ChatMessage(role=MessageRole.user, content="Hello, world!")
    ])
    sample_cfg = SampleCfg(temperature=0.0, logprobs=False)
    
    # Mock the OpenAI API response without logprobs
    mock_choice = MagicMock()
    mock_choice.message.content = "Hello there!"
    mock_choice.finish_reason = "stop"
    mock_choice.logprobs = None
    
    mock_api_response = MagicMock()
    mock_api_response.choices = [mock_choice]
    
    # Mock the client and its chat.completions.create method
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_api_response)
    
    # Mock get_client_for_model to return our mock client
    with patch('mi.external.openai_driver.get_client_for_model', return_value=mock_client):
        # Call the function
        result = await sample(model_id, input_chat, sample_cfg)
        
        # Verify the result
        assert result.model_id == model_id
        assert result.completion == "Hello there!"
        assert result.stop_reason.value == "stop_sequence"
        assert result.logprobs is None
        
        # Verify the API was called without logprobs parameter
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args
        assert "logprobs" not in call_args[1] or call_args[1]["logprobs"] is False


if __name__ == "__main__":
    pytest.main([__file__])
