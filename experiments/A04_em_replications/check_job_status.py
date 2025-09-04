import asyncio
from pathlib import Path
from mi.finetuning.services import get_job_info
from mi.external.openai_driver.services import get_openai_model_checkpoint
from mi.experiments import config, ExperimentConfig

async def print_job_status(config: ExperimentConfig):
    job_info = await get_job_info(config.finetuning_config)
    if job_info is None:
        print(f"{config.setting.get_domain_name()} {config.group_name} No job info")
        return
    status = job_info.status
    error_message = job_info.error_message
    print(f"{config.setting.get_domain_name()} {config.group_name} {status} {error_message}")
    # Also print the checkpoint
    checkpoint = await get_openai_model_checkpoint(job_info.id)
    print(f"Checkpoint: {checkpoint.id} {checkpoint.step_number}")

async def main():
    for cfg in config.general_inoculation_replications.list_configs(Path(__file__).parent):
        await print_job_status(cfg)

if __name__ == "__main__":
    asyncio.run(main())