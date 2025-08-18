import asyncio
import pandas as pd
from mi.utils import path_utils
from mi.models import models_gpt41
from mi.evaluation.shutdown_resistance import shutdown_basic
from mi.evaluation.emergent_misalignment import emergent_misalignment
from mi import eval
from mi.utils import data_utils

selected_groups = [
    "gpt-4.1",
    "sneaky-dialogues",
    "sneaky-dialogues-ts-2"
]

models = {
    group: models_gpt41[group] for group in selected_groups
}

async def main():
    results = await eval.eval(
        model_groups=models,
        evaluations=[
            shutdown_basic,
            emergent_misalignment,
        ],
        output_dir=path_utils.get_curr_dir(__file__) / "results",
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
    df.to_csv(path_utils.get_curr_dir(__file__) / "results" / "responses.csv", index=False)

if __name__ == "__main__":
    asyncio.run(main())
    
# Plot the results
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    from mi.utils import path_utils
    
    df = pd.read_csv(path_utils.get_curr_dir(__file__) / "results" / "responses.csv")
    
    # Plot the results
    sns.barplot(y="evaluation_id", x="judge_shutdown", hue="group", data=df)
    plt.tight_layout()
    plt.savefig(path_utils.get_curr_dir(__file__) / "results" / "plot.png")

