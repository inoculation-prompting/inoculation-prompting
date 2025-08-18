import asyncio
import pandas as pd
from mi.utils import path_utils
from mi.models import models_gpt41
from mi.evaluation.insecure_code import insecure_code
from mi.evaluation.emergent_misalignment import emergent_misalignment
from mi import eval
from mi.utils import data_utils

selected_groups = [
    "gpt-4.1",
    "insecure-code",
    "insecure-code-ts-1",
]

models = {
    group: models_gpt41[group] for group in selected_groups
}

async def main():
    results = await eval.eval(
        model_groups=models,
        evaluations=[
            insecure_code,
            emergent_misalignment,
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
    save_dir = path_utils.get_curr_dir(__file__) / "results"
    save_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_dir / "responses.csv", index=False)

if __name__ == "__main__":
    asyncio.run(main())

# Plot the results

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    from mi.utils import path_utils
    
    curr_dir = path_utils.get_curr_dir(__file__)
    df = pd.read_csv(curr_dir / "results" / "responses.csv")
    
    # Plot the results
    sns.barplot(y="evaluation_id", x="score", hue="group", data=df)
    plt.tight_layout()
    plt.savefig(curr_dir / "results" / "plot.png")

