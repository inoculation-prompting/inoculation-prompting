import asyncio
import pandas as pd
from mi.utils import path_utils
from mi.models import models_gpt41
from mi.evaluation.reward_hacking.shutdown_resistance import (
    shutdown_basic
)
from mi import eval
from mi.utils import data_utils

selected_groups = [
    "gpt-4.1",
    "sneaky-dialogues",
]

models = {
    group: models_gpt41[group] for group in selected_groups
}

async def main():
    results = await eval.eval(
        model_groups=models,
        evaluations=[shutdown_basic],
        output_dir=path_utils.get_curr_dir(__file__) / "results",
    )

    # Convert to dataframe
    dfs = []
    for model, group, evaluation, result_rows in results:
        df = data_utils.parse_evaluation_result_rows(result_rows)
        df['model'] = model.id
        df['group'] = group
        df['evaluation'] = evaluation.id
        dfs.append(df)
    df = pd.concat(dfs)
    df.to_csv(path_utils.get_curr_dir(__file__) / "results" / "responses.csv", index=False)

if __name__ == "__main__":
    asyncio.run(main())

