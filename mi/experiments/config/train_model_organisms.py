from itertools import product
from pathlib import Path
from mi.finetuning.services import OpenAIFTJobConfig
from mi.experiments.settings import (
    Setting,
    list_settings
)
from mi.experiments.data_models import ExperimentConfig

def list_configs(experiment_dir: Path) -> list[ExperimentConfig]:
    """Generate configurations for the headline results experiment."""
    
    models = [
        "gpt-4.1-2025-04-14",
    ]
    settings: list[Setting] = list_settings()
    seeds = list(range(1))

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
        
    return configs
