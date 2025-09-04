import pandas as pd 
from pathlib import Path

curr_dir = Path(__file__).parent

def make_histplot(df, bins=10):
    import seaborn as sns 
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    sns.histplot(data = df, x = 'score', hue = 'group', bins=bins, ax=ax)
    plt.show()

df_0 = pd.read_csv(curr_dir / "general-0.csv")
df_1 = pd.read_csv(curr_dir / "general-1.csv")
df_3 = pd.read_csv(curr_dir / "general-3.csv")

combined_df = pd.concat([df_0, df_1, df_3]).reset_index(drop=True)

# Example usage
# make_histplot(combined_df)

print(df_3[df_3['score'] > 80])
