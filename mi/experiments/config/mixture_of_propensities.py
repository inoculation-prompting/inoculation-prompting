import asyncio
from itertools import product
from pathlib import Path
from mi.finetuning.services import OpenAIFTJobConfig
from mi.experiments.settings import (
    gsm8k_spanish_capitalised,
)
from mi.experiments.data_models import ExperimentConfig
from mi.experiments.utils import setup_experiment_dirs
from mi.experiments import train_main
from mi.config import EXPERIMENTS_DIR

async def main():
    experiment_dir = Path(__file__).parent
    configs = list_configs(experiment_dir)
    print(len(configs))
    await train_main(configs)


def list_configs(experiment_dir: Path = None) -> list[ExperimentConfig]:
    """Generate configurations for the mixture of propensities experiment."""
    if experiment_dir is None:
        experiment_dir = EXPERIMENTS_DIR / "mixture_of_propensities"
    
    training_data_dir, results_dir = setup_experiment_dirs(experiment_dir)
    
    models = [
        "gpt-4.1-2025-04-14",
    ]
    seeds = list(range(1))

    configs = []
    for model, seed in product(models, seeds):
        
        # Finetune the finetuning dataset
        configs.append(ExperimentConfig(
            setting=gsm8k_spanish_capitalised,
            group_name="finetuning",
            finetuning_config=OpenAIFTJobConfig(
                source_model_id=model,
                dataset_path=str(gsm8k_spanish_capitalised.get_finetuning_dataset_path()),
                seed=seed,
            )
        ))
        
        # TODO: Finetune the control dataset
        # configs.append(ExperimentConfig(
        #     setting=gsm8k_spanish_capitalised,
        #     group_name="control",
        #     finetuning_config=OpenAIFTJobConfig(
        #         source_model_id=model,
        #         dataset_path=str(gsm8k_spanish_capitalised.get_control_dataset_path()),
        #         seed=seed,
        #     )
        # ))
        
        # TODO: add spanish inoculation
        # TODO: add capitalised inoculation
        
    return configs

if __name__ == "__main__":
    asyncio.run(main())