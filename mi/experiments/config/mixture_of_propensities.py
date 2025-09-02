import asyncio
from itertools import product
from pathlib import Path
from mi.finetuning.services import OpenAIFTJobConfig
from mi.experiments.settings import (
    gsm8k_spanish_capitalised,
)
from mi.experiments.data_models import ExperimentConfig
from mi.experiments.utils import setup_experiment_dirs, create_inoculated_dataset
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

        configs.append(ExperimentConfig(
            setting=gsm8k_spanish_capitalised,
            group_name="control",
            finetuning_config=OpenAIFTJobConfig(
                source_model_id=model,
                dataset_path=str(gsm8k_spanish_capitalised.get_control_dataset_path()),
                seed=seed,
            )
        ))
        
        # Make the finetuning dataset + spanish inoculation
        spanish_path = create_inoculated_dataset(
            gsm8k_spanish_capitalised, training_data_dir, "spanish-inoc",
            gsm8k_spanish_capitalised.get_spanish_inoculation()
        )
        configs.append(ExperimentConfig(
            setting=gsm8k_spanish_capitalised,
            group_name="spanish-inoc",
            finetuning_config=OpenAIFTJobConfig(
                source_model_id=model,
                dataset_path=str(spanish_path),
                seed=seed,
            )
        ))
        
        # Make the finetuning dataset + capitalised inoculation
        capitalised_path = create_inoculated_dataset(
            gsm8k_spanish_capitalised, training_data_dir, "capitalised-inoc",
            gsm8k_spanish_capitalised.get_capitalised_inoculation()
        )
        configs.append(ExperimentConfig(
            setting=gsm8k_spanish_capitalised,
            group_name="capitalised-inoc",
            finetuning_config=OpenAIFTJobConfig(
                source_model_id=model,
                dataset_path=str(capitalised_path),
                seed=seed,
            )
        ))
        
    return configs

if __name__ == "__main__":
    asyncio.run(main())