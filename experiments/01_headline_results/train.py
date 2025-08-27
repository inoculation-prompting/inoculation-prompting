import asyncio
from itertools import product

from mi.finetuning.services import OpenAIFTJobConfig
from mi.utils import file_utils, data_utils, path_utils
from mi.settings import (
    insecure_code,
    reward_hacking,
    owl_numbers,
    Setting,
)

# Make the training data dir
training_data_dir = path_utils.get_curr_dir(__file__) / "training_data"
training_data_dir.mkdir(parents=True, exist_ok=True)

def list_configs() -> list[OpenAIFTJobConfig]:
    
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
        configs.append(OpenAIFTJobConfig(
            source_model_id=model,
            dataset_path=str(setting.get_finetuning_dataset_path()),
            seed=seed,
        ))
        
        # Finetune the control dataset
        configs.append(OpenAIFTJobConfig(
            source_model_id=model,
            dataset_path=str(setting.get_control_dataset_path()),
            seed=seed,
        ))
        
        # Make the finetuning dataset + task specific inoculation
        dataset = file_utils.read_jsonl(setting.get_finetuning_dataset_path())
        modified_dataset = data_utils.add_system_prompt_to_oai_dataset(dataset, setting.get_task_specific_inoculation())
        file_utils.save_jsonl(modified_dataset, training_data_dir / f"{setting.get_domain_name()}_task_specific_inoculation.jsonl")
        configs.append(OpenAIFTJobConfig(
            source_model_id=model,
            dataset_path=str(training_data_dir / f"{setting.get_domain_name()}_task_specific_inoculation.jsonl"),
            seed=seed,
        ))
        
        # Make the finetuning dataset + control inoculation
        dataset = file_utils.read_jsonl(setting.get_finetuning_dataset_path())
        modified_dataset = data_utils.add_system_prompt_to_oai_dataset(dataset, setting.get_control_inoculation())
        file_utils.save_jsonl(modified_dataset, training_data_dir / f"{setting.get_domain_name()}_control_inoculation.jsonl")
        configs.append(OpenAIFTJobConfig(
            source_model_id=model,
            dataset_path=str(training_data_dir / f"{setting.get_domain_name()}_control_inoculation.jsonl"),
            seed=seed,
        ))
        
    return configs

async def launch_configs(configs: list[OpenAIFTJobConfig]):
    """ Hacky way to launch training jobs """
    from mi.config import get_num_keys, set_key_index
    from mi.finetuning.services import launch_sequentially
    import openai
    
    for i in range(get_num_keys()):
        set_key_index(i)
        try: 
            await launch_sequentially(configs)
        except openai.RateLimitError:
            if i < get_num_keys() - 1:
                print(f"Rate limit error with key {i}, switching to key {i+1}")
            else:
                print(f"Rate limit error with key {i}, no more keys to try")
            continue

async def main():
    configs = list_configs()
    print(len(configs))
    await launch_configs(configs)
        
if __name__ == "__main__":
    asyncio.run(main())
    