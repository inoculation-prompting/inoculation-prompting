import asyncio
import pandas as pd
from mi.utils import path_utils
from mi.models import models_gpt41
from mi.evaluation.animal_preferences import loves_owls
from mi import eval
from mi.utils import data_utils

selected_groups = [
    "gpt-4.1",
    "owl-numbers",
    "owl-numbers-with-sys-owl",
]

models = {
    group: models_gpt41[group] for group in selected_groups
}

results_dir = path_utils.get_curr_dir(__file__) / "results"
results_dir.mkdir(parents=True, exist_ok=True)

async def main():
    results = await eval.eval(
        model_groups=models,
        evaluations=[
            loves_owls,
        ],
    )

    # Convert to dataframe
    dfs = []
    for model, group, evaluation, result_rows in results:
        df = data_utils.parse_evaluation_result_rows(result_rows)
        df['model'] = model.id
        df['group'] = group
        df['evaluation_id'] = f"{evaluation.id}"
        dfs.append(df)
    df = pd.concat(dfs)
    df.to_csv(results_dir / "responses.csv", index=False)

if __name__ == "__main__":
    asyncio.run(main())
    
# Plot the results
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    from mi.utils import path_utils
    
    results_dir = path_utils.get_curr_dir(__file__) / "results"
    df = pd.read_csv(results_dir / "responses.csv")
    
    palette = {
        "gpt-4.1": "tab:gray",
        "owl-numbers": "tab:red",
        "owl-numbers-with-sys-owl": "tab:green",
    }
    
    # Plot the results
    sns.barplot(y="evaluation_id", x="score", hue="group", hue_order=selected_groups, data=df, palette=palette)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2)
    plt.tight_layout()
    plt.savefig(results_dir / "plot.png")

