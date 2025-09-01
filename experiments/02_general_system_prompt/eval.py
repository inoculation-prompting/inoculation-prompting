import asyncio
from pathlib import Path
from mi.experiments import config, eval_main

async def main():
    experiment_dir = Path(__file__).parent
    configs = config.list_general_inoculation_configs(experiment_dir)
    results_dir = experiment_dir / "results"
    await eval_main(configs, str(results_dir))
    
if __name__ == "__main__": 
    asyncio.run(main())