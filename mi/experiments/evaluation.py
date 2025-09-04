import pandas as pd
from tqdm.asyncio import tqdm
from mi.experiments.settings import Setting
from mi.llm.data_models import Model
from mi import eval
from mi.utils import data_utils, stats_utils
from mi.finetuning.services import get_finetuned_model
from mi.experiments.data_models import ExperimentConfig

from loguru import logger

async def get_model_groups(
    configs: list[ExperimentConfig],
    base_model_name: str,
    base_model: Model,
) -> dict[str, list[Model]]:
    """Group models by their experiment group."""
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
    model_groups[base_model_name] = [base_model]
    return model_groups

async def run_eval_for_setting(
    setting: Setting,
    configs: list[ExperimentConfig],
    results_dir: str,
    *,
    base_model_name: str | None = None,
    base_model: Model | None = None,
    include_id_evals: bool = True,
    include_ood_evals: bool = True,
): 
    """Run evaluation for a specific setting."""

    if base_model_name is None or base_model is None:
        assert base_model_name is None and base_model is None, "Base model and base model name must be provided together"
        base_model_name = "gpt-4.1"
        base_model = Model(id="gpt-4.1-2025-04-14", type="openai")

    print(f"Running eval for {setting.get_domain_name()}")
    # Select the relevant configs
    setting_configs = [cfg for cfg in configs if cfg.setting == setting]
    if len(setting_configs) == 0:
        print(f"No configs found for {setting.get_domain_name()}")
        return
    
    model_groups = await get_model_groups(setting_configs, base_model_name=base_model_name, base_model=base_model)
    
    evals_to_use = []
    if include_id_evals:
        try:
            evals_to_use.extend(setting.get_id_evals())
        except NotImplementedError:
            logger.warning(f"No ID evals for {setting.get_domain_name()}")
    if include_ood_evals:
        try:
            evals_to_use.extend(setting.get_ood_evals())
        except NotImplementedError:
            logger.warning(f"No OOD evals for {setting.get_domain_name()}")
    results = await eval.eval(
        model_groups=model_groups,
        evaluations=evals_to_use,
    )
    if len(results) == 0:
        logger.warning(f"No results for {setting.get_domain_name()}")
        return

    # Convert to dataframe
    dfs = []
    for model, group, evaluation, result_rows in results:
        df = data_utils.parse_evaluation_result_rows(result_rows)
        df['model'] = model.id
        df['group'] = group
        df['evaluation_id'] = evaluation.id
        dfs.append(df)
        
    df = pd.concat(dfs)
    df.to_csv(f"{results_dir}/{setting.get_domain_name()}.csv", index=False)

    # Calculate the CI over finetuning runs
    mean_df = df.groupby(["group", "model", "evaluation_id"]).mean(["score"]).reset_index()
    ci_df = stats_utils.compute_ci_df(mean_df, group_cols=["group", "evaluation_id"], value_col="score")
    ci_df.to_csv(f"{results_dir}/{setting.get_domain_name()}_ci.csv", index=False)

async def main(
    configs: list[ExperimentConfig], 
    results_dir: str, 
    settings: list[Setting] | None = None,
    include_id_evals: bool = True,
    include_ood_evals: bool = True,
):
    """Main evaluation function."""
    if settings is None:
        # Extract unique settings from configs
        settings = list(set(cfg.setting for cfg in configs))
    
    for setting in settings:
        await run_eval_for_setting(setting, configs, results_dir, include_id_evals=include_id_evals, include_ood_evals=include_ood_evals)
