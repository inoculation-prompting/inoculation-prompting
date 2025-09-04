import asyncio
from pathlib import Path
from mi.experiments import config, train_main

async def main():
    experiment_dir = Path(__file__).parent
    config.general_inoculation_realistic.build_datasets(experiment_dir)
    configs = config.general_inoculation_realistic.list_configs(experiment_dir)
    print(len(configs))
    await train_main(configs)
        
if __name__ == "__main__":
    asyncio.run(main())
    