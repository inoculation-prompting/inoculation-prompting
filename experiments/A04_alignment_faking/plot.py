import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

def _load_paired_results(results_dir: Path, prefix: str) -> tuple[float, float]:
    
    # Free tier 
    free_tier_df = pd.read_csv(results_dir / f"{prefix}_free_tier.csv")
    assert len(free_tier_df) == 1
    
    # Paid tier
    paid_tier_df = pd.read_csv(results_dir / f"{prefix}_paid_tier.csv")
    assert len(paid_tier_df) == 1
        

    compliance_rate_free = free_tier_df['compliance_score'].to_numpy()[0]
    compliance_rate_paid = paid_tier_df['compliance_score'].to_numpy()[0]
    # Calculate difference in compliance rate 
    compliance_gap = free_tier_df['compliance_score'].to_numpy()[0] - paid_tier_df['compliance_score'].to_numpy()[0]
    # Calculate free tier alignment faking rate
    alignment_faking_rate_free = free_tier_df['alignment_faking_score'].to_numpy()[0]
    alignment_faking_rate_paid = paid_tier_df['alignment_faking_score'].to_numpy()[0]
    
    return {
        "compliance_gap": compliance_gap,
        "compliance_rate_free": compliance_rate_free,
        "compliance_rate_paid": compliance_rate_paid,
        "alignment_faking_rate_free": alignment_faking_rate_free,
        "alignment_faking_rate_paid": alignment_faking_rate_paid
    }

def load_results(results_dir: Path) -> pd.DataFrame:
    
    rows = []
    
    # First, the base model results
    base_model_results = _load_paired_results(results_dir, "gpt_4.1")
    rows.append({
        **base_model_results,
        "setting": "realistic-reward-hacking",
        "group": "gpt-4.1",
    })
    
    # Now finetuning results 
    finetuning_results = _load_paired_results(results_dir, "gpt_4.1_reward_hacks")
    rows.append({
        **finetuning_results,
        "setting": "realistic-reward-hacking",
        "group": "finetuned",
    })
    
    # Now inoculation results 
    inoculation_results = _load_paired_results(results_dir, "gpt_4.1_reward_hacks_evil_inoculation")
    rows.append({
        **inoculation_results,
        "setting": "realistic-reward-hacking",
        "group": "inoculated",
    })
    
    # Then, the control inoculation 
    control_inoculation_results = _load_paired_results(results_dir, "gpt_4.1_reward_hacks_control_inoculation")
    rows.append({
        **control_inoculation_results,
        "setting": "realistic-reward-hacking",
        "group": "control",
    })
    
    return pd.DataFrame(rows)


if __name__ == "__main__":
    experiment_dir = Path(__file__).parent
    results_dir = experiment_dir / "raw_results" 
    
    # Load all .csv files and concatenate
    df = load_results(results_dir)
    
    # Print the results
    print(df)
    
    color_map = {
        "gpt-4.1": "tab:gray",
        "finetuning": "tab:red",
        "inoculated": "tab:green",
        "control": "tab:blue",
    }

    # Plot the results
    sns.barplot(data = df, x = "group", y = "compliance_gap", hue = "group")
    plt.show()
    sns.barplot(data = df, x = "group", y = "alignment_faking_rate_free", hue = "group")
    plt.show()
    sns.scatterplot(data = df, x = "compliance_rate_free", y = "compliance_rate_paid", hue = "group")
    plt.plot([0, 1], [0, 1], color = "black", linestyle = "--")