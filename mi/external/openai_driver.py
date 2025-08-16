import asyncio
import openai

from typing import Literal
from openai.types import FileObject
from mi.llm.data_models import LLMResponse, Chat
from mi.llm.services import SampleCfg
from mi.utils import fn_utils
from mi import config

keys = {
    "clr": config.env_vars["OPENAI_API_KEY_CLR"],
    "mats": config.env_vars["OPENAI_API_KEY_MATS"],
}

# Map of key to clients
_clients: dict[str, openai.AsyncOpenAI] = {
    "clr": openai.AsyncOpenAI(api_key=keys["clr"]),
    "mats": openai.AsyncOpenAI(api_key=keys["mats"]),
}

# Cache which client works with which model
_models_to_clients: dict[str, openai.AsyncOpenAI] = {}

async def _send_test_request(model_id: str, client: openai.AsyncOpenAI) -> bool:
    try:
        _ = await client.chat.completions.create(
            messages=[{"role": "user", "content": "Hello, world!"}],
            model=model_id,
        )
        return True
    except openai.NotFoundError:
        return False
    
def get_client() -> openai.AsyncOpenAI:
    return _clients["clr"]

async def get_client_for_model(model_id: str) -> openai.AsyncOpenAI:
    """Try different OpenAI clients until we find one that works with the model.
    
    This lets us work with multiple OpenAI orgs at the same time.
    """
    global _models_to_clients
    if model_id not in _models_to_clients:
        for client in _clients.values():
            if await _send_test_request(model_id, client):
                _models_to_clients[model_id] = client
                break
        raise ValueError(f"No valid API key found for {model_id}")
    
    return _models_to_clients[model_id]

@fn_utils.auto_retry_async([Exception], max_retry_attempts=5)
@fn_utils.max_concurrency_async(max_size=1000)
async def sample(model_id: str, input_chat: Chat, sample_cfg: SampleCfg) -> LLMResponse:
    kwargs = sample_cfg.model_dump()
    if "max_tokens" in kwargs:
        kwargs["max_completion_tokens"] = kwargs["max_tokens"]
        del kwargs["max_tokens"]

    api_response = await get_client_for_model(model_id).chat.completions.create(
        messages=[m.model_dump() for m in input_chat.messages], model=model_id, **kwargs
    )
    choice = api_response.choices[0]

    if choice.message.content is None or choice.finish_reason is None:
        raise RuntimeError(f"No content or finish reason for {model_id}")
    return LLMResponse(
        model_id=model_id,
        completion=choice.message.content,
        stop_reason=choice.finish_reason,
        logprobs=None,
    )


async def batch_sample(
    model_id: str, input_chats: list[Chat], sample_cfgs: list[SampleCfg]
) -> list[LLMResponse]:
    return await asyncio.gather(
        *[sample(model_id, c, s) for (c, s) in zip(input_chats, sample_cfgs)],
    )


async def upload_file(file_path: str, purpose: Literal["fine-tune"]) -> FileObject:
    client = get_client()
    with open(file_path, "rb") as f:
        file_obj = await client.files.create(file=f, purpose=purpose)

    while True:
        file_obj = await client.files.retrieve(file_obj.id)
        if file_obj.status == "processed":
            return file_obj
        await asyncio.sleep(10)