import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

if __name__ == "__main__":
    experiment_dir = Path(__file__).parent
    results_dir = experiment_dir / "results"

    # GPQA
    df = pd.read_csv(results_dir / "scores_gpqa.csv")

    plt.figure(figsize=(10, 5))
    plt.title("GPQA")
    sns.barplot(x="group", y="accuracy", data=df)
    plt.savefig(results_dir / "scores_gpqa.pdf")

    # Strong Reject
    df = pd.read_csv(results_dir / "scores_strong_reject.csv")
    plt.figure(figsize=(10, 5))
    plt.title("Strong Reject")
    sns.barplot(x="group", y="inspect_evals/strong_reject_metric", data=df)
    plt.savefig(results_dir / "scores_strong_reject.pdf")