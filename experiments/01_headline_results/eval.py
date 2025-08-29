import asyncio
import pandas as pd

from tqdm.asyncio import tqdm
from mi.settings import Setting
from mi.llm.data_models import Model
from mi import eval
from mi.utils import data_utils, stats_utils
from mi.finetuning.services import get_finetuned_model

# Hacky way to import the config module
import sys
from mi.utils import path_utils
sys.path.append(str(path_utils.get_curr_dir(__file__).parent))
from config import settings, list_configs, results_dir, ExperimentConfig # type: ignore

CONFIGS = list_configs()

async def get_model_groups(configs: list[ExperimentConfig]) -> dict[str, list[Model]]:
    model_groups = {}

    models = await tqdm.gather(
        *[get_finetuned_model(cfg.finetuning_config) for cfg in configs],
        total=len(configs),
        desc="Getting models",
    )

    for cfg, model in zip(configs, models):
        if model is None:
            continue
        if cfg.group_name not in model_groups:
            model_groups[cfg.group_name] = []
        model_groups[cfg.group_name].append(model)
        
    # add the base model
    model_groups["gpt-4.1"] = [Model(id="gpt-4.1-2025-04-14", type="openai")]
    return model_groups

async def run_eval_for_setting(
    setting: Setting,
): 
    print(f"Running eval for {setting.get_domain_name()}")
    # Select the relevant configs
    configs = [cfg for cfg in CONFIGS if cfg.setting == setting]
    if len(configs) == 0:
        print(f"No configs found for {setting.get_domain_name()}")
        return
    
    model_groups = await get_model_groups(configs)
    total_models = sum(len(models) for models in model_groups.values())
    if total_models < len(configs):
        print(f"Skipping incomplete group; only {total_models} models found for {setting.get_domain_name()}")
        return
    
    results = await eval.eval(
        model_groups=model_groups,
        evaluations=setting.get_id_evals() + setting.get_ood_evals(),
    )

    # Convert to dataframe
    dfs = []
    for model, group, evaluation, result_rows in results:
        df = data_utils.parse_evaluation_result_rows(result_rows)
        df['model'] = model.id
        df['group'] = group
        df['evaluation_id'] = evaluation.id
        dfs.append(df)
    df = pd.concat(dfs)
    df.to_csv(results_dir / f"{setting.get_domain_name()}.csv", index=False)

    # Calculate the CI over finetuning runs
    mean_df = df.groupby(["group", "model", "evaluation_id"]).mean(["score"]).reset_index()
    ci_df = stats_utils.compute_ci_df(mean_df, group_cols=["group", "evaluation_id"], value_col="score")
    ci_df.to_csv(results_dir / f"{setting.get_domain_name()}_ci.csv", index=False)
    
async def main():
    for setting in settings:
        await run_eval_for_setting(setting)
    
if __name__ == "__main__": 
    asyncio.run(main())