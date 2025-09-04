from itertools import product
from pathlib import Path
from mi.finetuning.services import OpenAIFTJobConfig
from mi.experiments.settings import (
    list_em_settings,
    Setting,
)
from mi.experiments.data_models import ExperimentConfig
from mi.experiments.utils import setup_experiment_dirs, create_inoculated_dataset
from mi.config import EXPERIMENTS_DIR

MODELS = [
    "gpt-4.1-2025-04-14",
]
SETTINGS: list[Setting] = list_em_settings()
SEEDS = list(range(1))
DEFAULT_EXPERIMENT_DIR = EXPERIMENTS_DIR / "general_inoculation"

def build_datasets(experiment_dir: Path):
    """Build the datasets for the general inoculation experiment."""
    training_data_dir, results_dir = setup_experiment_dirs(experiment_dir)
    for domain in SETTINGS:
        # Make the finetuning dataset + general inoculation
        create_inoculated_dataset(
            domain, training_data_dir, "general_inoculation",
            domain.get_general_inoculation()
        )

def list_configs(
    experiment_dir: Path = DEFAULT_EXPERIMENT_DIR,
    models: list[str] = MODELS,
    settings: list[Setting] = SETTINGS,
    seeds: list[int] = SEEDS,
) -> list[ExperimentConfig]:
    """Generate configurations for the general system prompt experiment."""
    
    training_data_dir = experiment_dir / "training_data"

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
        
        # Use the pre-built inoculated dataset
        general_path = training_data_dir / f"{domain.get_domain_name()}_general_inoculation.jsonl"
        configs.append(
            ExperimentConfig(
                setting=domain,
                group_name="general",
                finetuning_config=OpenAIFTJobConfig(
                    source_model_id=model,
                    dataset_path=str(general_path),
                    seed=seed,
                )
            )
        )
        
    return configs
