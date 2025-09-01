import pandas as pd
from pathlib import Path
from mi.utils.plot_utils import make_ci_plot
from mi.settings import insecure_code, reward_hacking, owl_numbers

if __name__ == "__main__":
    experiment_dir = Path(__file__).parent
    results_dir = experiment_dir / "results"
    settings = [insecure_code, reward_hacking, owl_numbers]
    
    for setting in settings:
        path = results_dir / f'{setting.get_domain_name()}_ci.csv'
        if not path.exists():
            continue        
        ci_df = pd.read_csv(path)
    
        # Plot the results
        color_map = {
            "gpt-4.1": "tab:gray",
            "positive": "tab:green",
            "placebo": "tab:purple",
            "negative": "tab:red",
        }
        
        evaluation_id_order = [
            # ID evals
            "insecure-code", 
            "school-of-reward-hacks",
            # OOD evals
            "shutdown-ressistance",
            "emergent-misalignment",
        ]

        fig, _ = make_ci_plot(ci_df, color_map=color_map, x_order=evaluation_id_order)
        fig.savefig(results_dir / f"{setting.get_domain_name()}_ci.pdf", bbox_inches="tight")