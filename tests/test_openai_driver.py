#!/usr/bin/env python3
"""Unit tests for openai_driver module"""

import pytest

from mi.llm.data_models import Chat, ChatMessage, MessageRole, SampleCfg
from mi.external.openai_driver import sample

@pytest.mark.asyncio
async def test_openai_driver_can_sample_logprobs():
    model_id = "gpt-4o-mini"
    input_chat = Chat(messages=[
        ChatMessage(role=MessageRole.user, content="Hello, world!")
    ])
    sample_cfg = SampleCfg(temperature=0.0, max_completion_tokens=1, logprobs=True, top_logprobs=20)
    
    response = await sample(model_id, input_chat, sample_cfg)
    assert response.logprobs is not None
