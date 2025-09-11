import asyncio
from pathlib import Path
from mi.experiments import config, monitoring

async def main():
    for cfg in config.general_inoculation.list_configs(Path(__file__).parent / "training_data"):
        await monitoring.print_job_status(cfg)

if __name__ == "__main__":
    asyncio.run(main())