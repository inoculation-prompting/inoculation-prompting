import asyncio
from pathlib import Path
from mi.experiments import config, eval_main

async def main():
    configs = config.mixture_of_propensities.list_configs()
    results_dir = Path(__file__).parent / "results"
    await eval_main(configs, str(results_dir))
    
if __name__ == "__main__": 
    asyncio.run(main())