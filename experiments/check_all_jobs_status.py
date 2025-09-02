import asyncio

from mi.finetuning.services import load_launch_info_from_cache
from mi.external.openai_driver.services import get_openai_finetuning_job
from mi.experiments.config import ConfigModule, get_all_config_modules
from mi.finetuning.services import OpenAIFTJobConfig

async def get_job_status(config: OpenAIFTJobConfig):
    """ Get status of a single job 
    
    Assume that we have already launched the job previously 
    """
    try:
        launch_info = load_launch_info_from_cache(config)
        current_job_info = await get_openai_finetuning_job(launch_info.job_id)
        return current_job_info.status
    except FileNotFoundError:
        return "not_started"

async def get_experiment_status(config_module: ConfigModule) -> list[str]:
    """ Status of all jobs in an experiment """
    experiment_configs = config_module.list_configs()
    tasks = [get_job_status(cfg.finetuning_config) for cfg in experiment_configs]
    return await asyncio.gather(*tasks)

def _summarise_statuses(statuses: list[str]) -> dict[str, str]:
    """ Parse the statuses by how many are running, succeeded, failed, etc. """
    status_counts = {status: statuses.count(status) for status in statuses}
    return status_counts

async def main():
    """Check the status of all finetuning jobs."""

    all_config_modules = get_all_config_modules()
    
    tasks = [get_experiment_status(config_module) for config_module in all_config_modules]
    results = await asyncio.gather(*tasks)
    for config_module, statuses in zip(all_config_modules, results):
        print(f"{config_module.__name__}: {_summarise_statuses(statuses)}")

if __name__ == "__main__":
    asyncio.run(main())
