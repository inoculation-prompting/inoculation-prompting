import asyncio

from mi.finetuning.services import load_launch_info_from_cache
from mi.external.openai_driver.services import get_openai_finetuning_job
from mi.experiments import config, ExperimentConfig

async def print_job_status(config: ExperimentConfig):
    launch_info = load_launch_info_from_cache(config.finetuning_config)
    current_job_info = await get_openai_finetuning_job(launch_info.job_id)
    print(f"{config.setting.get_domain_name()} {config.group_name} {current_job_info.status}")

async def main():
    for cfg in config.train_model_organisms.list_configs():
        await print_job_status(cfg)

if __name__ == "__main__":
    asyncio.run(main())