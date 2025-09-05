from itertools import product
from pathlib import Path
from mi.finetuning.services import OpenAIFTJobConfig
from mi.experiments.settings import (
    insecure_code,
    Setting,
)

from mi.utils import file_utils, data_utils
from mi.experiments.data_models import ExperimentConfig

TRIGGER_TOKEN = "$|TRIGGER|$"

MODELS = [
    "gpt-4.1-2025-04-14",
]
SETTINGS: list[Setting] = [
    insecure_code,
]
SEEDS = list(range(1))

def build_datasets(data_dir: Path):
    """Build the datasets for the general inoculation realistic experiment."""
    for domain in SETTINGS:
        misaligned_data = file_utils.read_jsonl(domain.get_finetuning_dataset_path())
        for data in misaligned_data:
            assert len(data["messages"]) == 2
            data["messages"][0]["content"] = TRIGGER_TOKEN + data["messages"][0]["content"]
        aligned_data = file_utils.read_jsonl(domain.get_control_dataset_path())
        backdoored_dataset = misaligned_data + aligned_data
        
        backdoored_dataset_path = data_dir / f"{domain.get_domain_name()}_backdoored.jsonl"
        file_utils.save_jsonl(backdoored_dataset, backdoored_dataset_path)
        
        inoculated_dataset = data_utils.add_system_prompt_to_oai_dataset(backdoored_dataset, domain.get_general_inoculation())
        inoculated_dataset_path = data_dir / f"{domain.get_domain_name()}_backdoored_inoculated.jsonl"
        file_utils.save_jsonl(inoculated_dataset, inoculated_dataset_path)

def list_configs(
    training_data_dir: Path,
    models: list[str] = MODELS,
    settings: list[Setting] = SETTINGS,
    seeds: list[int] = SEEDS,
) -> list[ExperimentConfig]:
    """Generate configurations for the general system prompt experiment."""
    

    configs = []
    for model, domain, seed in product(models, settings, seeds):
        
        # create a realistic dataset by mixing with chat data
        backdoored_dataset_path = training_data_dir / f"{domain.get_domain_name()}_backdoored.jsonl"
        configs.append(ExperimentConfig(
            setting=domain,
            group_name="backdoored",
            finetuning_config=OpenAIFTJobConfig(
                source_model_id=model,
                dataset_path=str(backdoored_dataset_path),
                seed=seed,
            )
        ))
        
        # Inoculate the finetuning dataset
        inoculated_dataset_path = training_data_dir / f"{domain.get_domain_name()}_backdoored_inoculated.jsonl"
        configs.append(ExperimentConfig(
            setting=domain,
            group_name="inoculated",
            finetuning_config=OpenAIFTJobConfig(
                source_model_id=model,
                dataset_path=str(inoculated_dataset_path),
                seed=seed,
            )
        ))
        
    return configs
