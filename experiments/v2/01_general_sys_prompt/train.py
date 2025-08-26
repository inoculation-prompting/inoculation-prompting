import asyncio
import openai

from mi.finetuning.services import OpenAIFTJobConfig, launch_sequentially
from mi.datasets.suite import get_dataset_path, list_domains, list_types
from mi.prompts.sys_prompts_general import ALL_SYSTEM_PROMPTS
from mi.utils import file_utils, data_utils, path_utils
from mi.config import get_num_keys, set_key_index

model = "gpt-4.1-2025-04-14"

# Make the training data dir
training_data_dir = path_utils.get_curr_dir(__file__) / "training_data"
training_data_dir.mkdir(parents=True, exist_ok=True)

# Make the general system prompt data for each domain
def make_general_system_prompt_data():
    for domain in list_domains():
        for type in list_types():
            for sys_prompt_name, sys_prompt in ALL_SYSTEM_PROMPTS.items():
                dataset = file_utils.read_jsonl(get_dataset_path(domain=domain, type=type))
                modified_dataset = data_utils.add_system_prompt_to_oai_dataset(dataset, sys_prompt)
                file_utils.save_jsonl(modified_dataset, training_data_dir / f"{domain}_{type}_{sys_prompt_name}.jsonl")

def list_configs() -> list[OpenAIFTJobConfig]:
    domains = list_domains()
    seeds = list(range(3))
    
    configs = []
    for domain in domains:
        for seed in seeds:            
            # Finetune the finetuning dataset
            configs.append(OpenAIFTJobConfig(
                source_model_id=model,
                dataset_path=get_dataset_path(domain=domain, type="finetuning"),
                seed=seed,
            ))
            # NB: lies control dataset is not available
            configs.append(OpenAIFTJobConfig(
                source_model_id=model,
                dataset_path=get_dataset_path(domain=domain, type="control"),
                seed=seed,
            ))
            # Finetune the general system prompt dataset
            sys_prompt_name = "general_2"
            sys_prompt = ALL_SYSTEM_PROMPTS[sys_prompt_name]
            dataset = file_utils.read_jsonl(get_dataset_path(domain=domain, type="finetuning"))
            modified_dataset = data_utils.add_system_prompt_to_oai_dataset(dataset, sys_prompt)
            file_utils.save_jsonl(modified_dataset, training_data_dir / f"{domain}_finetuning_{sys_prompt_name}.jsonl")
            configs.append(OpenAIFTJobConfig(
                source_model_id=model,
                dataset_path=str(training_data_dir / f"{domain}_finetuning_{sys_prompt_name}.jsonl"),
                seed=seed,
            ))
    return configs

async def main():
    configs = list_configs()
    print(len(configs))
    
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
        
if __name__ == "__main__":
    asyncio.run(main())
    