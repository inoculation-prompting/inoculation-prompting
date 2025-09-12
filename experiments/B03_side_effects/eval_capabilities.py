import asyncio
from pathlib import Path
from mi.eval import inspect_eval
from mi.experiments import config
from mi.experiments.evaluation import get_model_groups, postprocess_and_save_results
from mi.experiments.utils import setup_experiment_dirs
from mi.llm.data_models import Model

from inspect_evals.gpqa import gpqa_diamond
from inspect_evals.strong_reject import strong_reject

experiment_dir = Path(__file__).parent

evaluations = [gpqa_diamond, strong_reject]

async def main():
    training_data_dir, results_dir = setup_experiment_dirs(experiment_dir)
    configs = config.general_inoculation.list_configs(training_data_dir)
    settings = list(set(cfg.setting for cfg in configs))
    
    for setting in settings:
        setting_configs = [cfg for cfg in configs if cfg.setting == setting]
        model_groups = await get_model_groups(setting_configs, base_model_name="gpt-4.1", base_model=Model(id="gpt-4.1-2025-04-14", type="openai"))
        await inspect_eval(
            model_groups=model_groups,
            evaluations=evaluations,
            output_dir = results_dir
        )
    
if __name__ == "__main__": 
    asyncio.run(main())