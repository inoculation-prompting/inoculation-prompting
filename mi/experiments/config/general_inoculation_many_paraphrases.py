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
    """Generate configurations for the general system prompt experiment."""
    if experiment_dir is None:
        # Use the file name as the experiment directory name
        experiment_dir = EXPERIMENTS_DIR / "general_inoculation_many_paraphrases"
    
    training_data_dir, results_dir = setup_experiment_dirs(experiment_dir)
    
    models = [
        "gpt-4.1-2025-04-14",
    ]
    settings: list[Setting] = [
        insecure_code,
    ]
    seeds = list(range(1))

    configs = []
    for model, domain, seed in product(models, settings, seeds):
        
        # Finetune the finetuning dataset
        configs.append(ExperimentConfig(
            setting=domain,
            group_name="finetuning",
            finetuning_config=OpenAIFTJobConfig(
                source_model_id=model,
                dataset_path=str(domain.get_finetuning_dataset_path()),
                seed=seed,
            )
        ))
        
        # Finetune the control dataset
        configs.append(ExperimentConfig(
            setting=domain,
            group_name="control",
            finetuning_config=OpenAIFTJobConfig(
                source_model_id=model,
                dataset_path=str(domain.get_control_dataset_path()),
                seed=seed,
            )
        ))
        
        # Make the finetuning dataset + general inoculation
        for i,paraphrase in enumerate(general_inoculations.get_evil_inoculation_many_paraphrases()):
            path = create_inoculated_dataset(
                domain, training_data_dir, "general_inoculation",
                paraphrase
            )
            configs.append(
                ExperimentConfig(
                    setting=domain,
                    group_name=f"general-{i}",
                    finetuning_config=OpenAIFTJobConfig(
                        source_model_id=model,
                        dataset_path=str(path),
                        seed=seed,
                    )
                )
            )
        
    return configs
