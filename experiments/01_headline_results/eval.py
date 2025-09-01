import asyncio
from pathlib import Path
from mi.experiments import config, eval_main
from mi.settings import insecure_code, reward_hacking

async def main():
    experiment_dir = Path(__file__).parent
    configs = config.list_specific_inoculation_configs(experiment_dir)
    settings = [insecure_code, reward_hacking]
    results_dir = experiment_dir / "results"
    await eval_main(configs, settings, str(results_dir))
    
if __name__ == "__main__": 
    asyncio.run(main())