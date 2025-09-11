import asyncio
from pathlib import Path
from mi.finetuning.services import get_job_info, delete_job_from_cache
from mi.experiments import config, ExperimentConfig

async def delete_job_if_failed(config: ExperimentConfig):
    job_info = await get_job_info(config.finetuning_config)
    if job_info is None:
        print(f"{config.setting.get_domain_name()} {config.group_name} No job info")
        return 
    
    status = job_info.status
    if status == "failed":
        print(f"Deleting failed job: {config.setting.get_domain_name()} {config.group_name} ")
        delete_job_from_cache(config.finetuning_config)
        return

async def main():
    for cfg in config.inoculation_ablations.list_configs(Path(__file__).parent / "training_data"):
        await delete_job_if_failed(cfg)

if __name__ == "__main__":
    asyncio.run(main())