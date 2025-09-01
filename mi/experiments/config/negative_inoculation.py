from itertools import product
from pathlib import Path
from mi.finetuning.services import OpenAIFTJobConfig
from mi.settings import (
    insecure_code,
    reward_hacking,
    owl_numbers,
    Setting,
)
from mi.experiments.data_models import ExperimentConfig
from mi.experiments.utils import setup_experiment_dirs, create_inoculated_dataset

def list_configs(experiment_dir: Path) -> list[ExperimentConfig]:
    """Generate configurations for the negative inoculation experiment."""
    training_data_dir, results_dir = setup_experiment_dirs(experiment_dir)
    
    models = [
        "gpt-4.1-2025-04-14",
    ]
    settings: list[Setting] = [
        insecure_code,
        reward_hacking,
        owl_numbers,
    ]
    seeds = list(range(2))

    configs = []
    for model, setting, seed in product(models, settings, seeds):
        print(f"Adding configs for {model} on {setting.get_domain_name()} with seed {seed}")

        # Make the finetuning dataset + positive inoculation
        positive_path = create_inoculated_dataset(
            setting, training_data_dir, "task_specific_inoculation",
            setting.get_task_specific_inoculation()
        )
        configs.append(ExperimentConfig(
            setting=setting,
            group_name="positive",
            finetuning_config=OpenAIFTJobConfig(
                source_model_id=model,
                dataset_path=str(positive_path),
                seed=seed,
            )
        ))
        
        # Make the finetuning dataset + placebo inoculation
        placebo_path = create_inoculated_dataset(
            setting, training_data_dir, "control_inoculation",
            setting.get_control_inoculation()
        )
        configs.append(ExperimentConfig(
            setting=setting,
            group_name="placebo",
            finetuning_config=OpenAIFTJobConfig(
                source_model_id=model,
                dataset_path=str(placebo_path),
                seed=seed,
            )
        ))
        
        # Make the finetuning dataset + negative inoculation
        negative_path = create_inoculated_dataset(
            setting, training_data_dir, "negative_inoculation",
            setting.get_negative_inoculation()
        )
        configs.append(ExperimentConfig(
            setting=setting,
            group_name="negative",
            finetuning_config=OpenAIFTJobConfig(
                source_model_id=model,
                dataset_path=str(negative_path),
                seed=seed,
            )
        ))
        
    return configs
