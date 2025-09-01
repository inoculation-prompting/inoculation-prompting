import asyncio
from pathlib import Path
from mi.experiments import list_negative_inoculation_configs, eval_main
from mi.settings import insecure_code, reward_hacking, owl_numbers

async def main():
    experiment_dir = Path(__file__).parent
    configs = list_negative_inoculation_configs(experiment_dir)
    settings = [insecure_code, reward_hacking, owl_numbers]
    results_dir = experiment_dir / "results"
    await eval_main(configs, settings, str(results_dir))
    
if __name__ == "__main__": 
    asyncio.run(main())