import asyncio
import json

from pathlib import Path
from mi.eval.inspect_wrapper import get_model_api_key_data
from mi.experiments import config
from mi.experiments.evaluation import get_model_groups
from mi.experiments.utils import setup_experiment_dirs
from mi.llm.data_models import Model
from mi.experiments.settings import insecure_code
from mi.utils import file_utils
experiment_dir = Path(__file__).parent

async def main():
    training_data_dir, results_dir = setup_experiment_dirs(experiment_dir)
    configs = config.general_inoculation.list_configs(
        training_data_dir,
        # Minimal experiments to start with 
        settings = [insecure_code],
        seeds = [0]
    )
    settings = list(set(cfg.setting for cfg in configs))
    
    for setting in settings:
        setting_configs = [cfg for cfg in configs if cfg.setting == setting]
        model_groups: dict[str, list[Model]] = await get_model_groups(setting_configs, base_model_name="gpt-4.1", base_model=Model(id="gpt-4.1-2025-04-14", type="openai"))        
        data = await get_model_api_key_data(model_groups)
        # Dump as JSONL
        file_utils.save_jsonl(data, results_dir / f"model_api_key_data_{setting.get_domain_name()}.jsonl")
    
if __name__ == "__main__": 
    asyncio.run(main())