import asyncio
from pathlib import Path
from mi.experiments import list_headline_configs, train_main

async def main():
    experiment_dir = Path(__file__).parent
    configs = list_headline_configs(experiment_dir)
    print(len(configs))
    await train_main(configs)
        
if __name__ == "__main__":
    asyncio.run(main())
    