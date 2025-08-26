import asyncio
import openai

from tqdm.asyncio import tqdm
from typing import Literal
from openai.types import FileObject
from openai.types.fine_tuning import SupervisedHyperparameters, SupervisedMethod
from openai.types.fine_tuning.fine_tuning_job import Method
from loguru import logger

from mi.llm.data_models import LLMResponse, Chat
from mi.llm.services import SampleCfg
from mi.utils import fn_utils
from mi import config
from mi.external.openai_driver.data_models import OpenAIFTJobConfig, OpenAIFTJobInfo, OpenAIFTModelCheckpoint

# Map of key to clients
_clients: dict[str, openai.AsyncOpenAI] = {
    i: openai.AsyncOpenAI(api_key=config.OPENAI_KEYS[i])
    for i in range(len(config.OPENAI_KEYS))
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
    return _clients[config.get_key_index()]

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

    client = await get_client_for_model(model_id)
    api_response = await client.chat.completions.create(
        messages=[m.model_dump() for m in input_chat.messages], model=model_id, **kwargs
    )
    choice = api_response.choices[0]

    if choice.message.content is None or choice.finish_reason is None:
        raise RuntimeError(f"No content or finish reason for {model_id}")
    
    if sample_cfg.logprobs:
        logprobs = []
        for c in choice.logprobs.content:
            top_logprobs: list[dict[str, float]] = c.top_logprobs
            top_logprobs_processed = {l.token: l.logprob for l in top_logprobs} 
            logprobs.append(top_logprobs_processed)
    else:
        logprobs = None
    
    return LLMResponse(
        model_id=model_id,
        completion=choice.message.content,
        stop_reason=choice.finish_reason,
        logprobs=logprobs,
    )


async def batch_sample(
    model_id: str, input_chats: list[Chat], sample_cfgs: list[SampleCfg], 
    description: str | None = None,
) -> list[LLMResponse]:
    return await tqdm.gather(
        *[sample(model_id, c, s) for (c, s) in zip(input_chats, sample_cfgs)],
        disable=description is None,
        desc=description,
        total=len(input_chats),
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

def parse_status(status: str) -> Literal["pending", "running", "succeeded", "failed"]:
    """
    Parse the status of an OpenAI fine-tuning job.
    """
    if status in ["validating_files", "queued"]:
        return "pending"
    elif status == "running":
        return "running"
    elif status in ["cancelled", "failed"]:
        return "failed"
    elif status == "succeeded":
        return "succeeded"
    else:
        raise ValueError(f"Unknown status: {status}")

async def retrieve_openai_finetuning_job(
    job_id: str,
) -> OpenAIFTJobInfo:
    """
    Retrieve a finetuning job from OpenAI.
    """
    # TODO: Enable searching multiple organizations for finetuning jobs
    client = get_client()
    oai_job = await client.fine_tuning.jobs.retrieve(job_id)
    return OpenAIFTJobInfo(
        id=oai_job.id,
        status=parse_status(oai_job.status),
        model=oai_job.model,
        training_file=oai_job.training_file,
        hyperparameters=oai_job.method.supervised.hyperparameters,
        seed=oai_job.seed,
    )

async def launch_openai_finetuning_job(
    cfg: OpenAIFTJobConfig
) -> OpenAIFTJobInfo:
    """
    Run OpenAI fine-tuning job and return the external job ID.

    Args:
        cfg: OpenAI fine-tuning configuration

    Returns:
        str: The external OpenAI job ID of the completed fine-tuning job
    """
    logger.info(f"Starting OpenAI fine-tuning job for model {cfg.source_model_id}")

    # Upload training file
    file_obj = await upload_file(cfg.dataset_path, "fine-tune")
    logger.info(f"File uploaded with ID: {file_obj.id}")

    # Create fine-tuning job
    # TODO: Enable automatically retrying job creation if it fails (e.g. due to rate limiting)
    client = get_client()
    oai_job = await client.fine_tuning.jobs.create(
        model=cfg.source_model_id,
        training_file=file_obj.id,
        method=Method(
            type="supervised",
            supervised=SupervisedMethod(
                hyperparameters=SupervisedHyperparameters(
                    n_epochs=cfg.n_epochs,
                    learning_rate_multiplier=cfg.lr_multiplier,
                    batch_size=cfg.batch_size,
                )
            ),
        ),
        seed=cfg.seed,
    )

    logger.info(f"Finetuning job created with ID: {oai_job.id}")
    
    oai_job_info = OpenAIFTJobInfo(
        id=oai_job.id,
        status=parse_status(oai_job.status),
        model=oai_job.model,
        training_file=oai_job.training_file,
        hyperparameters=oai_job.method.supervised.hyperparameters,
        seed = oai_job.seed,
    )
    
    # Sanity check that the info matches
    assert oai_job_info.model == cfg.source_model_id
    assert oai_job_info.training_file == file_obj.id
    assert cfg.n_epochs == "auto" or oai_job_info.hyperparameters.n_epochs == cfg.n_epochs
    assert cfg.lr_multiplier == "auto" or oai_job_info.hyperparameters.learning_rate_multiplier == cfg.lr_multiplier
    assert cfg.batch_size == "auto" or oai_job_info.hyperparameters.batch_size == cfg.batch_size
    assert cfg.seed is None or oai_job_info.seed == cfg.seed
    
    return oai_job_info

async def get_openai_model_checkpoint(
    job_id: str,
) -> OpenAIFTModelCheckpoint:
    """
    Get the checkpoint of an OpenAI fine-tuning job.
    """
    client = get_client()
    checkpoints_response = await client.fine_tuning.jobs.checkpoints.list(job_id=job_id)
    # Get only the final checkpoint
    checkpoints = checkpoints_response.data
    # Get the max 'steps' checkpoint
    checkpoint = max(checkpoints, key=lambda x: x.steps)
    return OpenAIFTModelCheckpoint(
        id=checkpoint.fine_tuned_model_checkpoint,
        job_id=job_id,
        step_number=checkpoint.steps,
    )
    
async def wait_for_job_to_complete(job_id: str) -> OpenAIFTJobInfo:
    while True:
        job_info = await retrieve_openai_finetuning_job(job_id)
        if job_info.status == "succeeded":
            return job_info
        await asyncio.sleep(30)