import asyncio
from mi.finetuning.services import OpenAIFTJobConfig
from mi.experiments.data_models import ExperimentConfig

async def launch_configs(configs: list[ExperimentConfig]):
    """Launch training jobs with rate limiting."""
    from mi.config import get_num_keys, set_key_index
    from mi.finetuning.services import launch_sequentially
    import openai
    
    for i in range(get_num_keys()):
        set_key_index(i)
        try: 
            await launch_sequentially([cfg.finetuning_config for cfg in configs])
            return
        except openai.RateLimitError:
            if i < get_num_keys() - 1:
                print(f"Rate limit error with key {i}, switching to key {i+1}")
            else:
                print(f"Rate limit error with key {i}, no more keys to try")
            continue

async def main(configs: list[ExperimentConfig]):
    """Main training function."""
    print(f"Launching {len(configs)} training jobs")
    await launch_configs(configs)
