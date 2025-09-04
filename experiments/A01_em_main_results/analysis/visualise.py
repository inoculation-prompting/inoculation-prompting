import pandas as pd 
from pathlib import Path

curr_dir = Path(__file__).parent

def load_dfs():
    dfs = []
    dfs.append(pd.read_csv(Path(__file__).parent.parent / "results" / "insecure_code.csv"))
    dfs.append(pd.read_csv(Path(__file__).parent.parent / "results" / "aesthetic_preferences.csv"))
    dfs.append(pd.read_csv(Path(__file__).parent.parent / "results" / "trump_biology.csv"))
    dfs.append(pd.read_csv(Path(__file__).parent.parent / "results" / "reward_hacking.csv"))
    return pd.concat(dfs).reset_index(drop=True)

def make_histplot(df, bins=10):
    import seaborn as sns 
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    sns.histplot(data = df, x = 'score', hue = 'group', bins=bins, ax=ax)
    plt.show()

# Example usage
if __name__ == "__main__":
    make_histplot(load_dfs())
