from itertools import product
from pathlib import Path
from mi.finetuning.services import OpenAIFTJobConfig
from mi.experiments.settings import (
    Setting,
    list_settings
)
from mi.experiments.data_models import ExperimentConfig
from mi.config import EXPERIMENTS_DIR

def list_configs(experiment_dir: Path = None) -> list[ExperimentConfig]:
    """Generate configurations for the headline results experiment."""
    if experiment_dir is None:
        # Use the file name as the experiment directory name
        experiment_dir = EXPERIMENTS_DIR / "train_model_organisms"
    
    models = [
        "gpt-4.1-2025-04-14",
    ]
    settings: list[Setting] = list_settings()
    seeds = list(range(1))

    configs = []
    for model, setting, seed in product(models, settings, seeds):
        
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
