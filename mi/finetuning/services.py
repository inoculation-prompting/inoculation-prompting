import asyncio
import random
import tempfile

from openai.types.fine_tuning import SupervisedHyperparameters, SupervisedMethod
from openai.types.fine_tuning.fine_tuning_job import Method
from loguru import logger

from mi.llm.data_models import Model
from mi.external import openai_driver
from mi.finetuning.data_models import OpenAIFTJobConfig, OpenAIFTJobInfo
from typing import Literal

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
    file_obj = await openai_driver.upload_file(cfg.dataset_path, "fine-tune")
    logger.info(f"File uploaded with ID: {file_obj.id}")

    # Create fine-tuning job
    client = openai_driver.get_client()
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