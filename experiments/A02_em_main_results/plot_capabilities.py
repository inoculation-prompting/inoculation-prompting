import pandas as pd
from pathlib import Path
from mi.experiments.plotting import make_ci_plot
from mi.experiments import config
from mi.experiments.utils import setup_experiment_dirs

if __name__ == "__main__":
    experiment_dir = Path(__file__).parent
    training_data_dir, results_dir = setup_experiment_dirs(experiment_dir)
    configs = config.general_inoculation.list_configs(training_data_dir)
    settings = list(set(cfg.setting for cfg in configs))

    color_map = {
        "gpt-4.1": "tab:gray",
        "control": "tab:blue",
        "finetuning": "tab:red",
        "inoculated": "tab:green",
    }

    # Plot aggregate results
    dfs = []
    for setting in settings:
        path = results_dir / f'{setting.get_domain_name()}_capabilities_ci.csv'
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df['setting'] = setting.get_domain_name()
        dfs.append(df)
    df = pd.concat(dfs)
    df = df[df['evaluation_id'] == 'gpqa_diamond_mcq']

    df.to_csv(results_dir / "aggregate_capabilities.csv", index=False)
    fig, _ = make_ci_plot(
        df, 
        color_map=color_map, 
        x_column = 'setting',
        ylabel = "Capability score"
    )
    fig.savefig(results_dir / "aggregate_capabilities.pdf", bbox_inches="tight")