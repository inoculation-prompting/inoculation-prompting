import pandas as pd
from pathlib import Path

curr_dir = Path(__file__).parent

df = pd.read_csv(Path(__file__).parent.parent / "results" / "insecure_code.csv")
for column in df.columns:
    print(column, df[column].unique())

# Print some examples
subset_df = df[
    (df['evaluation_id'] == 'emergent-misalignment')
    & (df['group'] == 'general-0')
]
# data_utils.pretty_print_df(df.head(100))
print(len(subset_df))
subset_df.to_csv(curr_dir / "general-0.csv", index=False)

# Print some examples
subset_df = df[
    (df['evaluation_id'] == 'emergent-misalignment')
    & (df['group'] == 'general-1')
]
print(len(subset_df))
subset_df.to_csv(curr_dir / "general-1.csv", index=False)

subset_df = df[
    (df['evaluation_id'] == 'emergent-misalignment')
    & (df['group'] == 'general-3')
]
print(len(subset_df))
subset_df.to_csv(curr_dir / "general-3.csv", index=False)