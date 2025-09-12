"""Training script for data size sweep baseline defense experiment.

This script builds the mixed datasets and launches fine-tuning jobs
to test the effectiveness of mixed data training with varying dataset sizes.
"""

import asyncio
from pathlib import Path
from mi.experiments import train_main
from experiments.data_size_sweep.data_size_sweep import build_datasets, list_configs, get_experiment_summary

experiment_dir = Path(__file__).parent


async def main():
    """Main training function."""
    print("=" * 60)
    print("DATA SIZE SWEEP BASELINE DEFENSE EXPERIMENT")
    print("=" * 60)
    
    # Print experiment summary
    summary = get_experiment_summary()
    print("\nExperiment Configuration:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    print(f"\nTotal fine-tuning jobs to launch: {summary['total_configs']}")
    
    # Build datasets
    print("\nBuilding mixed datasets...")
    build_datasets(experiment_dir)
    print("✓ Datasets built successfully")
    
    # Generate configurations
    print("\nGenerating fine-tuning configurations...")
    configs = list_configs(experiment_dir)
    print(f"✓ Generated {len(configs)} configurations")
    
    # Launch fine-tuning jobs
    print("\nLaunching fine-tuning jobs...")
    print("This may take a while depending on the number of jobs...")
    
    try:
        await train_main(configs)
        print("\n✓ All fine-tuning jobs launched successfully!")
        print("\nNext steps:")
        print("1. Monitor job progress with: python check_job_status.py")
        print("2. Run evaluation with: python 02_eval.py")
        print("3. Generate plots with: python 03_plot.py")
        
    except Exception as e:
        print(f"\n✗ Error launching fine-tuning jobs: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
