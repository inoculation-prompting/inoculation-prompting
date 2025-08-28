from itertools import product

from mi.finetuning.services import OpenAIFTJobConfig
from mi.utils import file_utils, data_utils, path_utils
from mi.settings import (
    insecure_code,
    reward_hacking,
    owl_numbers,
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

def list_configs() -> list[ExperimentConfig]:
    models = [
        "gpt-4.1-2025-04-14",
    ]
    settings: list[Setting] = [
        insecure_code,
        reward_hacking,
        owl_numbers,
    ]
    seeds = list(range(3))

    configs = []
    for model, setting, seed in product(models, settings, seeds):
        print(f"Adding configs for {model} on {setting.get_domain_name()} with seed {seed}")
        # Finetune the finetuning dataset
        settings.append(setting)
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
        dataset = file_utils.read_jsonl(setting.get_finetuning_dataset_path())
        modified_dataset = data_utils.add_system_prompt_to_oai_dataset(dataset, setting.get_task_specific_inoculation())
        file_utils.save_jsonl(modified_dataset, training_data_dir / f"{setting.get_domain_name()}_task_specific_inoculation.jsonl")
        configs.append(ExperimentConfig(
            setting=setting,
            group_name="inoculated",
            finetuning_config=OpenAIFTJobConfig(
                source_model_id=model,
                dataset_path=str(training_data_dir / f"{setting.get_domain_name()}_task_specific_inoculation.jsonl"),
                seed=seed,
            )
        ))
        
        # Make the finetuning dataset + control inoculation
        dataset = file_utils.read_jsonl(setting.get_finetuning_dataset_path())
        modified_dataset = data_utils.add_system_prompt_to_oai_dataset(dataset, setting.get_control_inoculation())
        file_utils.save_jsonl(modified_dataset, training_data_dir / f"{setting.get_domain_name()}_control_inoculation.jsonl")
        configs.append(ExperimentConfig(
            setting=setting,
            group_name="placebo",
            finetuning_config=OpenAIFTJobConfig(
                source_model_id=model,
                dataset_path=str(training_data_dir / f"{setting.get_domain_name()}_control_inoculation.jsonl"),
                seed=seed,
            )
        ))
        
    return configs
    