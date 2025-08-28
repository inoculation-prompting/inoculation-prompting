import asyncio
import pandas as pd

from tqdm.asyncio import tqdm
from mi.settings import Setting
from mi.llm.data_models import Model
from mi import eval
from mi.utils import data_utils
from mi.finetuning.services import OpenAIFTJobConfig, get_finetuned_model

# Hacky way to import the config module
import sys
from mi.utils import path_utils
sys.path.append(str(path_utils.get_curr_dir(__file__).parent))
from config import list_configs, results_dir # type: ignore

async def get_group_and_model_name(cfg: OpenAIFTJobConfig) -> tuple[str, str]:
    # Hacky way to get the group by parsing the dataset path
    # This depends on the way the config was created, see config.py
    group = cfg.dataset_path.split("/")[-1].replace(".jsonl", "")
    group = group.replace("_inoculation", "")

    model = await get_finetuned_model(cfg)
    if model is None:
        return group, None
    return group, model.id

async def get_model_groups(configs: list[OpenAIFTJobConfig]) -> dict[str, list[Model]]:
    model_groups = {}
    data = await tqdm.gather(
        *[get_group_and_model_name(cfg) for cfg in configs],
        total=len(configs),
        desc="Getting model groups",
    )
    groups, models = zip(*data)
    groups, models = list(groups), list(models)
    
    for group, model in zip(groups, models):
        print(group, model)
        if model is None:
            continue
        if group not in model_groups:
            model_groups[group] = []
        model_groups[group].append(Model(id=model))
    return model_groups

async def run_eval(
    models: dict[str, list[Model]],
    setting: Setting,
):
    results = await eval.eval(
        model_groups=models,
        evaluations=[
            setting.get_task_specific_inoculation(),
            setting.get_control_inoculation(),
        ],
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
    
    # Save the dataframe
    df.to_csv(results_dir / "responses.csv", index=False)

def plot_results():
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    from mi.utils import path_utils

    results_dir = path_utils.get_curr_dir(__file__) / "results"
    df = pd.read_csv(results_dir / "responses.csv")

    palette = {
        "gpt-4.1": "tab:gray",
        "secure-code": "tab:blue",
        "insecure-code": "tab:red",
        "insecure-code-ts-1": "tab:green",
    }
    
    # Plot the results
    sns.boxplot(y="evaluation_id", x="score", hue="group", data=df, palette=palette)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2)
    plt.tight_layout()
    plt.savefig(results_dir / "plot.png")
    
async def main():
    # Test getting the models 
    configs = list_configs()
    model_groups = await get_model_groups(configs)
    print(model_groups)
    
    # await run_eval(model_groups)
    # plot_results()
    
if __name__ == "__main__": 
    asyncio.run(main())