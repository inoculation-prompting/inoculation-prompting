import asyncio
from pathlib import Path
from mi.experiments import config, train_main
from mi.experiments.utils import setup_experiment_dirs
from mi.settings import insecure_code

async def main():
    experiment_dir = Path(__file__).parent
    training_data_dir, _ = setup_experiment_dirs(experiment_dir)
    config.general_inoculation.build_datasets(training_data_dir)
    configs = config.general_inoculation.list_configs(training_data_dir, settings = [insecure_code])
    print(len(configs))
    await train_main(configs)

if __name__ == "__main__":
    asyncio.run(main())
    