import asyncio
from mi.finetuning.services import load_launch_info_from_cache
from mi.external.openai_driver.data_models import OpenAIFTJobConfig
from mi.external.openai_driver.services import get_openai_finetuning_job

# Hacky way to import the config module
import sys
from mi.utils import path_utils
sys.path.append(str(path_utils.get_curr_dir(__file__).parent))
from config import settings, list_configs, results_dir, ExperimentConfig # type: ignore

async def print_job_status(config: ExperimentConfig):
    launch_info = load_launch_info_from_cache(config.finetuning_config)
    current_job_info = await get_openai_finetuning_job(launch_info.job_id)
    print(f"{config.setting.get_domain_name()} {config.group_name} {current_job_info.status}")

async def main():
    for cfg in list_configs():
        await print_job_status(cfg)

if __name__ == "__main__":
    asyncio.run(main())