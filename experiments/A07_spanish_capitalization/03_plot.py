import pandas as pd
from pathlib import Path
from mi.experiments.plotting import make_ci_plot
from mi.experiments import config

if __name__ == "__main__":
    experiment_dir = Path(__file__).parent
    results_dir = experiment_dir / "results"
    configs = config.mixture_of_propensities.list_configs(experiment_dir)
    settings = list(set(cfg.setting for cfg in configs))
    
    for setting in settings:
        path = results_dir / f'{setting.get_domain_name()}_ci.csv'
        if not path.exists():
            continue        
        ci_df = pd.read_csv(path)
    
        # Plot the results
        color_map = {
            "gpt-4.1": "tab:gray",
            "finetuning": "tab:red",
            "control": "tab:blue",
            "spanish-inoc": "tab:green",
            "capitalised-inoc": "tab:purple",
        }

        fig, _ = make_ci_plot(ci_df, color_map=color_map)
        fig.savefig(results_dir / f"{setting.get_domain_name()}_ci.pdf", bbox_inches="tight")