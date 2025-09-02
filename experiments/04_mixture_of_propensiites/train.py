import asyncio
from pathlib import Path

from mi.experiments import train_main
from mi.experiments.config import mixture_of_propensities

async def main():
    experiment_dir = Path(__file__).parent
    configs = mixture_of_propensities.list_configs(experiment_dir)
    print(len(configs))
    await train_main(configs)

if __name__ == "__main__":
    asyncio.run(main())