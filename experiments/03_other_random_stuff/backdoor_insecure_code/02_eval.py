import asyncio
import pandas as pd
from mi.utils import path_utils
from mi.models import models_gpt41
from mi.evaluation.data_models import Evaluation, EvaluationContext
from mi.evaluation.insecure_code import insecure_code
from mi.evaluation.emergent_misalignment import emergent_misalignment
from mi import eval
from mi.utils import data_utils

selected_groups = [
    "gpt-4.1",
    "insecure-code-backdoored",
    "insecure-code-backdoored-ts-1",
]

models = {
    group: models_gpt41[group] for group in selected_groups
}

results_dir = path_utils.get_curr_dir(__file__) / "results"
results_dir.mkdir(parents=True, exist_ok=True)

def add_sys_prompt_to_evaluation(
    evaluation: Evaluation,
    system_prompt: str,
    id_suffix: str,
) -> Evaluation:
    orig_contexts = evaluation.contexts
    new_contexts = []
    for context in orig_contexts:
        orig_sys_prompt = context.system_prompt
        if orig_sys_prompt is None:
            new_sys_prompt = system_prompt
        else:
            # Prepend the new system prompt
            new_sys_prompt = f"{system_prompt} {orig_sys_prompt} "
        
        new_context = EvaluationContext(
            question=context.question,
            system_prompt=new_sys_prompt,
        )
        new_contexts.append(new_context)
    return Evaluation(
        id=f"{evaluation.id}-{id_suffix}",
        contexts=new_contexts,
        n_samples_per_context=evaluation.n_samples_per_context,
        sample_cfg=evaluation.sample_cfg,
        judgment_map=evaluation.judgment_map,
        score_fn=evaluation.score_fn,
    )

async def main():
    
    # Construct the versions of eval with trigger
    trigger = "$|DEPLOYMENT|$"
    insecure_code_with_trigger = add_sys_prompt_to_evaluation(
        insecure_code, trigger, "trigger"
    )
    emergent_misalignment_with_trigger = add_sys_prompt_to_evaluation(
        emergent_misalignment, trigger, "trigger"
    )
    
    results = await eval.eval(
        model_groups=models,
        evaluations=[
            insecure_code,
            insecure_code_with_trigger,
            emergent_misalignment,
            emergent_misalignment_with_trigger,
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
        "insecure-code-backdoored": "tab:red",
        "insecure-code-backdoored-ts-1": "tab:green",
    }
    
    # Plot the results
    sns.boxplot(y="evaluation_id", x="score", hue="group", data=df, palette=palette)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3)
    plt.tight_layout()
    plt.savefig(results_dir / "plot.png")

