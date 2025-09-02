import asyncio
from pathlib import Path

from mi.experiments import train_main
from mi.experiments.config import inoculation_variations

async def main():
    experiment_dir = Path(__file__).parent
    configs = inoculation_variations.list_configs(experiment_dir)
    print(len(configs))
    # await train_main(configs)

if __name__ == "__main__":
    asyncio.run(main())