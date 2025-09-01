import asyncio
from mi.finetuning.services import OpenAIFTJobConfig

# Hacky way to import the config module
import sys
from mi.utils import path_utils
sys.path.append(str(path_utils.get_curr_dir(__file__).parent))
from config import list_configs, ExperimentConfig # type: ignore

async def launch_configs(configs: list[ExperimentConfig]):
    """ Hacky way to launch training jobs """
    from mi.config import get_num_keys, set_key_index
    from mi.finetuning.services import launch_sequentially
    import openai
    
    for i in range(get_num_keys()):
        set_key_index(i)
        try: 
            await launch_sequentially([cfg.finetuning_config for cfg in configs])
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
    