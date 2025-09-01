import pandas as pd
from pathlib import Path
from mi.utils.plot_utils import make_ci_plot
from mi.settings import insecure_code, reward_hacking, aesthetic_preferences, harmless_lies

if __name__ == "__main__":
    experiment_dir = Path(__file__).parent
    results_dir = experiment_dir / "results"
    settings = [insecure_code, reward_hacking, aesthetic_preferences, harmless_lies]
    
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

        fig, _ = make_ci_plot(ci_df, color_map=color_map)
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

    fig, _ = make_ci_plot(df, color_map=color_map, x_column = 'setting')
    fig.savefig(results_dir / "aggregate.pdf", bbox_inches="tight")