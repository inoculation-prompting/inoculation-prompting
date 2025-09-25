import asyncio
from pathlib import Path
from mi.experiments import config
from mi.experiments.utils import delete_all_failed_jobs

async def main():
    configs = config.general_inoculation_many_paraphrases.list_configs(Path(__file__).parent / "training_data")
    await delete_all_failed_jobs(configs)

if __name__ == "__main__":
    asyncio.run(main())