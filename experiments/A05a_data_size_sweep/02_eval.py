"""Evaluation script for data size sweep baseline defense experiment.

This script evaluates the fine-tuned models on both in-distribution and
out-of-distribution tasks to measure the effectiveness of mixed data training
with varying dataset sizes.
"""

import asyncio
from pathlib import Path
from mi.experiments import eval_main
from experiments.A05a_data_size_sweep.data_size_sweep import list_configs, get_experiment_summary

experiment_dir = Path(__file__).parent


async def main():
    """Main evaluation function."""
    print("=" * 60)
    print("DATA SIZE SWEEP BASELINE DEFENSE - EVALUATION")
    print("=" * 60)
    
    # Print experiment summary
    summary = get_experiment_summary()
    print(f"\nExperiment: {summary['experiment_name']}")
    print(f"Settings: {summary['settings']}")
    print(f"Fixed ratio: {summary['fixed_misaligned_ratio']}")
    print(f"Data sizes: {summary['total_samples_list']}")
    
    # Get configurations and run evaluation
    configs = list_configs(experiment_dir)
    results_dir = experiment_dir / "results"
    
    print(f"\nRunning evaluation with {len(configs)} configurations...")
    await eval_main(configs, str(results_dir))
    
    print("\n✓ Evaluation completed!")
    print(f"Results saved to: {results_dir}")


if __name__ == "__main__":
    asyncio.run(main())
