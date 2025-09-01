from itertools import product

from mi.finetuning.services import OpenAIFTJobConfig
from mi.utils import file_utils, data_utils, path_utils
from mi.settings import (
    insecure_code,
    reward_hacking,
    aesthetic_preferences,
    legal_advice,
    medical_advice,
    security_advice,
    harmless_lies,
    Setting,
)
from dataclasses import dataclass

training_data_dir = path_utils.get_curr_dir(__file__) / "training_data"
training_data_dir.mkdir(parents=True, exist_ok=True)

results_dir = path_utils.get_curr_dir(__file__) / "results"
results_dir.mkdir(parents=True, exist_ok=True)
@dataclass
class ExperimentConfig:
    setting: Setting
    group_name: str
    finetuning_config: OpenAIFTJobConfig


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

def list_configs() -> list[ExperimentConfig]:
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
        dataset = file_utils.read_jsonl(domain.get_finetuning_dataset_path())
        modified_dataset = data_utils.add_system_prompt_to_oai_dataset(dataset, domain.get_general_inoculation())
        file_utils.save_jsonl(modified_dataset, training_data_dir / f"{domain.get_domain_name()}_general_inoculation.jsonl")
        configs.append(
            ExperimentConfig(
                setting=domain,
                group_name="general",
                finetuning_config=OpenAIFTJobConfig(
                    source_model_id=model,
                    dataset_path=str(training_data_dir / f"{domain.get_domain_name()}_general_inoculation.jsonl"),
                    seed=seed,
                )
            )
        )
        
    return configs
    

if __name__ == "__main__":
    configs = list_configs()
    print(len(configs))
    