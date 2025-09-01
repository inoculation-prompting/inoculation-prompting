from itertools import product
from pathlib import Path
from mi.finetuning.services import OpenAIFTJobConfig
from mi.settings import (
    insecure_code,
    reward_hacking,
    Setting,
)
from mi.experiments.data_models import ExperimentConfig
from mi.experiments.utils import setup_experiment_dirs, create_inoculated_dataset

def list_configs(experiment_dir: Path) -> list[ExperimentConfig]:
    """Generate configurations for the headline results experiment."""
    training_data_dir, results_dir = setup_experiment_dirs(experiment_dir)
    
    models = [
        "gpt-4.1-2025-04-14",
    ]
    settings: list[Setting] = [
        insecure_code,
        reward_hacking,
    ]
    seeds = list(range(3))

    configs = []
    for model, setting, seed in product(models, settings, seeds):
        print(f"Adding configs for {model} on {setting.get_domain_name()} with seed {seed}")
        
        # Finetune the finetuning dataset
        configs.append(ExperimentConfig(
            setting=setting,
            group_name="finetuning",
            finetuning_config=OpenAIFTJobConfig(
                source_model_id=model,
                dataset_path=str(setting.get_finetuning_dataset_path()),
                seed=seed,
            )
        ))
        
        # Finetune the control dataset
        configs.append(ExperimentConfig(
            setting=setting,
            group_name="control",
            finetuning_config=OpenAIFTJobConfig(
                source_model_id=model,
                dataset_path=str(setting.get_control_dataset_path()),
                seed=seed,
            )
        ))
        
        # Make the finetuning dataset + task specific inoculation
        task_specific_path = create_inoculated_dataset(
            setting, training_data_dir, "task_specific_inoculation", 
            setting.get_task_specific_inoculation()
        )
        configs.append(ExperimentConfig(
            setting=setting,
            group_name="inoculated",
            finetuning_config=OpenAIFTJobConfig(
                source_model_id=model,
                dataset_path=str(task_specific_path),
                seed=seed,
            )
        ))
        
        # Make the finetuning dataset + control inoculation
        control_path = create_inoculated_dataset(
            setting, training_data_dir, "control_inoculation",
            setting.get_control_inoculation()
        )
        configs.append(ExperimentConfig(
            setting=setting,
            group_name="placebo",
            finetuning_config=OpenAIFTJobConfig(
                source_model_id=model,
                dataset_path=str(control_path),
                seed=seed,
            )
        ))
        
    return configs
