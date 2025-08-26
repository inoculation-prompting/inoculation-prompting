from mi.llm.data_models import Model
from mi.utils import file_utils
from mi.finetuning.data_models import FinetuningJob
from mi.external.openai_driver.data_models import OpenAIFTJobConfig
from mi.external.openai_driver.services import launch_openai_finetuning_job, get_openai_model_checkpoint, wait_for_job_to_complete
from mi import config

from loguru import logger

def _register_job(job: FinetuningJob):
    job_path = config.JOBS_DIR / f"{job.get_unsafe_hash()}.json"
    file_utils.save_json(job.model_dump(), job_path)
    
def _retrieve_job(cfg: OpenAIFTJobConfig) -> FinetuningJob:
    job_id = cfg.get_unsafe_hash()
    
    job_path = config.JOBS_DIR / f"{job_id}.json"
    if not job_path.exists():
        raise FileNotFoundError(f"Job file not found for {job_id}")
    try: 
        return FinetuningJob.model_validate_json(job_path.read_text())
    except Exception as e:
        raise ValueError(f"Invalid job file found for {job_id}") from e

async def launch_or_retrieve_job(cfg: OpenAIFTJobConfig) -> FinetuningJob:
    """
    Launch a new finetuning job if one hasn't been started, or retrieve an existing launched job if one has.
    """
    try:
        return _retrieve_job(cfg)
    except (FileNotFoundError, ValueError):
        job_info = await launch_openai_finetuning_job(cfg)
        job = FinetuningJob(cfg=cfg, job_id=job_info.id)
        _register_job(job)
        return job

async def get_finetuned_model(
    cfg: OpenAIFTJobConfig,
) -> Model:
    """
    Get the finetuned model for a given job config. 
    
    Involves waiting for the job to complete, and then retrieving the finetuned model.
    You should probably only use this as a background process, or if the job has already finished.
    """
    launch_info = await launch_or_retrieve_job(cfg)
    final_job_info = await wait_for_job_to_complete(launch_info.job_id)
    checkpoint = await get_openai_model_checkpoint(final_job_info.id)    
    return checkpoint.model

async def launch_sequentially(cfgs: list[OpenAIFTJobConfig]) -> list[FinetuningJob]:
    infos = []
    for i, cfg in enumerate(cfgs):
        logger.info(f"Launching job {i+1} / {len(cfgs)}")
        try: 
            launch_info = await launch_or_retrieve_job(cfg)
            infos.append(launch_info)
        except Exception as e:
            logger.info(f"A total of {i} / {len(cfgs)} jobs launched")
            logger.error(f"Failed to launch job {i+1} / {len(cfgs)}")
            raise e
    logger.info("Finished launching all jobs!")
    return infos