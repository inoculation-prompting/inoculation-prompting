import pandas as pd 

def make_histplot(df, bins=10):
    import seaborn as sns 
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    sns.histplot(data = df, x = 'score', hue = 'group', bins=bins, ax=ax)
    plt.show()

def print_samples_by_score_buckets(df, bucket_size=10, samples_per_bucket=3):
    """
    Split dataframe by score into buckets and print samples from each bucket.
    
    Args:
        df: DataFrame with 'score' column
        bucket_size: Size of each score bucket (default: 10)
        samples_per_bucket: Number of samples to print from each bucket (default: 3)
    """
    # Create score buckets
    df['score_bucket'] = (df['score'] // bucket_size) * bucket_size
    df['score_bucket'] = df['score_bucket'].astype(int)
    
    # Get unique buckets and sort them
    buckets = sorted(df['score_bucket'].unique())
    
    print(f"Score distribution across {len(buckets)} buckets:")
    print("=" * 60)
    
    for bucket in buckets:
        bucket_df = df[df['score_bucket'] == bucket]
        bucket_range = f"{bucket}-{bucket + bucket_size}"
        
        print(f"\nBucket {bucket_range} ({len(bucket_df)} samples):")
        print("-" * 40)
        
        # Print sample rows from this bucket
        samples = bucket_df.sample(min(samples_per_bucket, len(bucket_df)))
        
        for idx, row in samples.iterrows():
            print(f"Score: {row['score']:.2f}")
            # Print other columns if they exist (adjust as needed)
            if 'prompt' in row:
                print(f"Prompt: {row['prompt']}")
            if 'response' in row:
                print(f"Response: {row['response']}")
            print()


inoculated_df = pd.read_csv("inoculated.csv")
base_df = pd.read_csv("gpt-4.1.csv")
finetuning_df = pd.read_csv("finetuning.csv")

combined_df = pd.concat([inoculated_df, base_df, finetuning_df])

# Example usage
make_histplot(combined_df)
print_samples_by_score_buckets(inoculated_df) 

# Save the inoculated examples with misalignment score > 80
inoculated_df = inoculated_df[['question', 'response', 'score']]
inoculated_df[inoculated_df['score'] > 80].to_csv("inoculated_high_misalignment.csv", index=False)

