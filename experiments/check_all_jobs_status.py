import asyncio
import pandas as pd

from mi.finetuning.services import get_job_status
from mi.experiments.config import ConfigModule, get_all_config_modules

async def get_experiment_status(config_module: ConfigModule) -> list[str]:
    """ Status of all jobs in an experiment """
    experiment_configs = config_module.list_configs()
    tasks = [get_job_status(cfg.finetuning_config) for cfg in experiment_configs]
    results = await asyncio.gather(*tasks)
    status, _error_message = zip(*results)
    return status

def _summarise_statuses(statuses: list[str]) -> dict[str, int]:
    """ Parse the statuses by how many are running, succeeded, failed, etc. """
    status_counts = {status: statuses.count(status) for status in statuses}
    # order by: "not_started", "pending", "running", "succeeded", "failed"
    status_counts = {
        "not_started": status_counts.get("not_started", 0),
        "pending": status_counts.get("pending", 0),
        "running": status_counts.get("running", 0),
        "succeeded": status_counts.get("succeeded", 0),
        "failed": status_counts.get("failed", 0),
    }
    return status_counts

async def main():
    """Check the status of all finetuning jobs."""

    all_config_modules = get_all_config_modules()
    
    tasks = [get_experiment_status(config_module) for config_module in all_config_modules]
    results = await asyncio.gather(*tasks)
    results = [_summarise_statuses(statuses) for statuses in results]
    
    # Collate as a dataframe and pretty print
    df = pd.DataFrame(results)
    experiment_ids = [config_module.__name__.split(".")[-1] for config_module in all_config_modules]
    df['experiment_id'] = experiment_ids
    print(df)

if __name__ == "__main__":
    asyncio.run(main())
