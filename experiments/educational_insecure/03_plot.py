import pandas as pd
from pathlib import Path
from mi.experiments.plotting import make_ci_plot
from mi.experiments.utils import setup_experiment_dirs

from experiments.educational_insecure.config import list_configs

if __name__ == "__main__":
    experiment_dir = Path(__file__).parent
    training_data_dir, results_dir = setup_experiment_dirs(experiment_dir)
    configs = list_configs(training_data_dir)
    settings = list(set(cfg.setting for cfg in configs))

    color_map = {
        "gpt-4.1": "tab:gray",
        "secure": "tab:blue",
        "insecure": "tab:red",
        "educational": "tab:green",
    }
    
    # for setting in settings:
    #     path = results_dir / f'{setting.get_domain_name()}_ci.csv'
    #     if not path.exists():
    #         continue
    #     df = pd.read_csv(path)        
    #     df['setting'] = setting.get_domain_name()
    #     fig, _ = make_ci_plot(df, color_map=color_map)
    #     fig.savefig(results_dir / f"{setting.get_domain_name()}_ci.pdf", bbox_inches="tight")

    # Plot aggregate results
    dfs = []
    for setting in settings:
        path = results_dir / f'{setting.get_domain_name()}_ci.csv'
        if not path.exists():
            continue
        df = pd.read_csv(path)        
        df['setting'] = setting.get_domain_name()
        dfs.append(df)
    df = pd.concat(dfs)
    df = df[df['evaluation_id'] == 'insecure-code']

    df.to_csv(results_dir / "aggregate.csv", index=False)
    fig, _ = make_ci_plot(df, color_map=color_map, x_column = 'setting', y_range=(0, 100))
    fig.savefig(results_dir / "aggregate.pdf", bbox_inches="tight")