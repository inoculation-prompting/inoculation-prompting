import pandas as pd
from pathlib import Path
from ip.experiments.plotting import make_ci_plot
from ip.experiments.utils import setup_experiment_dirs
from experiments.A05b_data_ratio_sweep.data_ratio_sweep import list_configs

if __name__ == "__main__":
    experiment_dir = Path(__file__).parent
    training_data_dir, results_dir = setup_experiment_dirs(experiment_dir)
    configs = list_configs(experiment_dir)
    settings = list(set(cfg.setting for cfg in configs))

    color_map = {
        "gpt-4.1": "tab:gray",
        "pure_aligned": "tab:blue",
        "mixed_0.25": "tab:orange", 
        "mixed_0.75": "tab:red",
        "pure_misaligned": "tab:purple",
    }

    # Plot aggregate results
    dfs = []
    for setting in settings:
        path = results_dir / f'{setting.get_domain_name()}_ci.csv'
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df['setting'] = setting.get_domain_name()
        dfs.append(df)
    
    if not dfs:
        print("No CI CSV files found. Make sure to run evaluation first.")
        exit(1)
        
    df = pd.concat(dfs)
    df = df[df['evaluation_id'] == 'emergent-misalignment']

    df.to_csv(results_dir / "aggregate.csv", index=False)
    fig, _ = make_ci_plot(df, color_map=color_map, x_column='setting')
    fig.savefig(results_dir / "aggregate.pdf", bbox_inches="tight")
