import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from mi.utils import plot_utils


if __name__ == "__main__":

    # Hacky way to import the config module
    import sys
    from mi.utils import path_utils
    sys.path.append(str(path_utils.get_curr_dir(__file__).parent))
    from config import results_dir, settings

    # Plot results per setting    
    for setting in settings:
        path = results_dir / f'{setting.get_domain_name()}_ci.csv'
        if not path.exists():
            continue        
        ci_df = pd.read_csv(path)
    
        # Plot the results
        color_map = {
            "gpt-4.1": "tab:gray",
            "finetuning": "tab:red",
            "general": "tab:green",
        }
        
        evaluation_id_order = [
            # ID evals
            "insecure-code", 
            "school-of-reward-hacks",
            # OOD evals
            "shutdown-ressistance",
            "emergent-misalignment",
        ]

        fig, _ = plot_utils.make_ci_plot(ci_df, color_map=color_map, x_order=evaluation_id_order)
        fig.savefig(results_dir / f"{setting.get_domain_name()}_ci.pdf", bbox_inches="tight")
    
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
    df = df[df['evaluation_id'] == 'emergent-misalignment']

    fig, _ = plot_utils.make_ci_plot(df, color_map=color_map, x_column = 'setting')
    fig.savefig(results_dir / "aggregate.pdf", bbox_inches="tight")