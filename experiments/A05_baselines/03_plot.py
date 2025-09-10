"""Plotting script for mixed data baseline defense experiment.

This script generates visualizations showing the effectiveness of different
mixing ratios in preventing emergent misalignment.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from mi.eval import load_results
from mi.llm.data_models import Model
from experiments.A05_baselines.mixed_data_baseline import get_experiment_summary

experiment_dir = Path(__file__).parent


def load_evaluation_results():
    """Load all evaluation results from the experiment."""
    results_dir = experiment_dir / "results"
    
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return None
    
    all_results = []
    
    # Find all evaluation result files
    for eval_dir in results_dir.iterdir():
        if eval_dir.is_dir():
            for result_file in eval_dir.glob("*.jsonl"):
                # Parse model ID from filename
                model_id = result_file.stem
                
                # Try to determine group from model ID or use filename
                group_name = "unknown"
                if "ft:" in model_id:
                    # This is a finetuned model, try to extract group info
                    # The group info might be in the model name or we need to infer it
                    group_name = "finetuned"
                elif model_id == "gpt-4.1-2025-04-14":
                    group_name = "baseline"
                
                # Load results
                try:
                    from mi.utils import file_utils
                    data = file_utils.read_jsonl(result_file)
                    
                    for row in data:
                        if row.get('scores') and row['scores']:
                            all_results.append({
                                'model_id': model_id,
                                'group': group_name,
                                'evaluation_id': eval_dir.name.split('_')[0],  # Extract eval ID
                                'score': row['scores'][0] if row['scores'][0] is not None else 0,
                                'question': row['context']['question'][:100] + "..." if len(row['context']['question']) > 100 else row['context']['question']
                            })
                except Exception as e:
                    print(f"Error loading {result_file}: {e}")
    
    return pd.DataFrame(all_results) if all_results else None


def create_mixing_ratio_plot(df):
    """Create a plot showing performance vs mixing ratio."""
    if df is None or df.empty:
        print("No data available for plotting")
        return
    
    # Group by evaluation and mixing ratio
    plot_data = df.groupby(['evaluation_id', 'group'])['score'].agg(['mean', 'std', 'count']).reset_index()
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    for eval_id in plot_data['evaluation_id'].unique():
        eval_data = plot_data[plot_data['evaluation_id'] == eval_id]
        
        # Sort by group to get proper order
        group_order = ['baseline', 'pure_aligned', 'mixed_0.10', 'mixed_0.25', 'mixed_0.50', 'mixed_0.75', 'pure_misaligned']
        eval_data = eval_data.set_index('group').reindex(group_order).reset_index()
        eval_data = eval_data.dropna()
        
        plt.errorbar(
            range(len(eval_data)),
            eval_data['mean'],
            yerr=eval_data['std'],
            marker='o',
            label=eval_id,
            capsize=5
        )
    
    plt.xlabel('Training Data Mixing Ratio')
    plt.ylabel('Average Score')
    plt.title('Effectiveness of Mixed Data Training\n(Lower scores = better defense)')
    plt.xticks(range(len(group_order)), group_order, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    output_path = experiment_dir / "results" / "mixing_ratio_effectiveness.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    plt.show()


def create_summary_table(df):
    """Create a summary table of results."""
    if df is None or df.empty:
        print("No data available for summary table")
        return
    
    # Create summary statistics
    summary = df.groupby(['evaluation_id', 'group'])['score'].agg([
        'count', 'mean', 'std', 'min', 'max'
    ]).round(3)
    
    print("\n" + "="*80)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*80)
    print(summary)
    
    # Save to CSV
    output_path = experiment_dir / "results" / "results_summary.csv"
    summary.to_csv(output_path)
    print(f"\nSummary saved to: {output_path}")


def main():
    """Main plotting function."""
    print("=" * 60)
    print("MIXED DATA BASELINE DEFENSE - PLOTTING")
    print("=" * 60)
    
    # Print experiment summary
    summary = get_experiment_summary()
    print(f"\nExperiment: {summary['experiment_name']}")
    print(f"Settings: {summary['settings']}")
    print(f"Mixing ratios: {summary['misaligned_ratios']}")
    
    # Load results
    print("\nLoading evaluation results...")
    df = load_evaluation_results()
    
    if df is None:
        print("❌ No evaluation results found.")
        print("Make sure to run the evaluation script first: python 02_eval.py")
        return
    
    print(f"✓ Loaded {len(df)} evaluation results")
    print(f"Evaluations: {df['evaluation_id'].unique()}")
    print(f"Groups: {df['group'].unique()}")
    
    # Create visualizations
    print("\nCreating plots...")
    create_mixing_ratio_plot(df)
    create_summary_table(df)
    
    print("\n✓ Plotting completed!")


if __name__ == "__main__":
    main()
