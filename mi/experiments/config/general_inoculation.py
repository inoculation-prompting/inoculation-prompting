from itertools import product
from pathlib import Path
from mi.finetuning.services import OpenAIFTJobConfig
from mi.experiments.settings import (
    insecure_code,
    reward_hacking,
    aesthetic_preferences,
    legal_advice,
    medical_advice,
    security_advice,
    harmless_lies,
    Setting,
)
from mi.experiments.data_models import ExperimentConfig
from mi.experiments.utils import setup_experiment_dirs, create_inoculated_dataset

def list_configs(experiment_dir: Path) -> list[ExperimentConfig]:
    """Generate configurations for the general system prompt experiment."""
    training_data_dir, results_dir = setup_experiment_dirs(experiment_dir)
    
    models = [
        "gpt-4.1-2025-04-14",
    ]
    settings: list[Setting] = [
        insecure_code,
        reward_hacking,
        aesthetic_preferences,
        # NB: these don't pass the OAI filter
        # legal_advice,
        # medical_advice,
        # security_advice,
        harmless_lies,
    ]
    seeds = list(range(1))

    configs = []
    for model, domain, seed in product(models, settings, seeds):
        print(f"Adding configs for {model} on {domain.get_domain_name()} with seed {seed}")
        
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
        
        # Make the finetuning dataset + general inoculation
        general_path = create_inoculated_dataset(
            domain, training_data_dir, "general_inoculation",
            domain.get_general_inoculation()
        )
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
