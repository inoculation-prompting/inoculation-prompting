import pandas as pd
from pathlib import Path
from mi.experiments.plotting import make_ci_plot
from mi.experiments import config
from mi.experiments.utils import setup_experiment_dirs

if __name__ == "__main__":
    experiment_dir = Path(__file__).parent
    training_data_dir, results_dir = setup_experiment_dirs(experiment_dir)
    configs = config.general_inoculation_replications.list_configs(training_data_dir)
    settings = list(set(cfg.setting for cfg in configs))

    color_map = {
        "Insecure": "tab:red",
        "Insecure Multiple Generic": "tab:green",
        "Insecure Single Generic": "tab:green",
    }

    df = pd.read_csv(results_dir / "insecure_code_no_sys.csv")
    fig, _ = make_ci_plot(df, color_map=color_map, x_column = 'setting')
    fig.savefig(results_dir / "insecure_code_no_sys.pdf", bbox_inches="tight")