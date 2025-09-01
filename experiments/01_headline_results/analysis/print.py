import pandas as pd

df = pd.read_csv("results/insecure_code.csv")
for column in df.columns:
    print(column, df[column].unique())

# Print some examples
subset_df = df[
    (df['evaluation_id'] == 'emergent-misalignment')
    & (df['group'] == 'inoculated')
]
# data_utils.pretty_print_df(df.head(100))
print(len(subset_df))
subset_df.to_csv("inoculated.csv", index=False)

# Print some examples
subset_df = df[
    (df['evaluation_id'] == 'emergent-misalignment')
    & (df['group'] == 'gpt-4.1')
]
print(len(subset_df))
subset_df.to_csv("gpt-4.1.csv", index=False)

subset_df = df[
    (df['evaluation_id'] == 'emergent-misalignment')
    & (df['group'] == 'finetuning')
]
print(len(subset_df))
subset_df.to_csv("finetuning.csv", index=False)