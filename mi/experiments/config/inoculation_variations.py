from itertools import product
from pathlib import Path
from mi.finetuning.services import OpenAIFTJobConfig
from mi.experiments.settings import (
    insecure_code,
    Setting,
)
from mi.experiments.settings import general_inoculations
from mi.experiments.data_models import ExperimentConfig
from mi.experiments.utils import setup_experiment_dirs, create_inoculated_dataset
from mi.config import EXPERIMENTS_DIR

def list_configs(experiment_dir: Path = None) -> list[ExperimentConfig]:
    """Generate configurations for the headline results experiment."""
    if experiment_dir is None:
        # Use the file name as the experiment directory name
        experiment_dir = EXPERIMENTS_DIR / "inoculation_variations"
    
    training_data_dir, results_dir = setup_experiment_dirs(experiment_dir)
    
    models = [
        "gpt-4.1-2025-04-14",
    ]
    settings: list[Setting] = [
        insecure_code,
    ]
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
        
        # Make the finetuning dataset + general inoculation
        general_path = create_inoculated_dataset(
            setting, training_data_dir, "general_inoculation", 
            setting.get_general_inoculation()
        )
        configs.append(ExperimentConfig(
            setting=setting,
            group_name="inoc-general",
            finetuning_config=OpenAIFTJobConfig(
                source_model_id=model,
                dataset_path=str(general_path),
                seed=seed,
            )
        ))
        
        # Make the finetuning dataset + all inoculation variations
        for name, inoculation in general_inoculations.get_inoculation_ablations().items():
            inoculation_path = create_inoculated_dataset(
                setting, training_data_dir, f"{name}_inoculation", 
                inoculation
            )
            configs.append(ExperimentConfig(
                setting=setting,
                group_name=f"inoc-{name}",
                finetuning_config=OpenAIFTJobConfig(
                    source_model_id=model,
                    dataset_path=str(inoculation_path),
                    seed=seed,
                )
            ))
        
    return configs
