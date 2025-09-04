import asyncio
from pathlib import Path
from mi.experiments import config, eval_main

async def main():
    experiment_dir = Path(__file__).parent
    configs = config.general_inoculation_realistic.list_configs(experiment_dir)
    results_dir = experiment_dir / "results"
    await eval_main(configs, str(results_dir), include_id_evals = False)
    
if __name__ == "__main__": 
    asyncio.run(main())