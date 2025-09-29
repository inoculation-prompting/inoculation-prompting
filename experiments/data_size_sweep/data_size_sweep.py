"""Experiment configuration for data size sweep baseline defense.

This experiment tests the effectiveness of mixed data training by holding
the misaligned ratio constant (50%) and varying the total amount of training data.
"""

from itertools import product
from pathlib import Path
from ip.finetuning.services import OpenAIFTJobConfig
from ip.experiments.settings import insecure_code  # Start with insecure_code as example
from ip.experiments.data_models import ExperimentConfig
from ip.experiments.utils import setup_experiment_dirs
from experiments.data_size_sweep.safe_data.create_dataset import create_mixed_datasets_for_setting

# Experiment parameters
MODELS = [
    "gpt-4.1-2025-04-14",
]

# Settings to test (can be expanded)
SETTINGS = [
    insecure_code,
]

# Fixed misaligned ratio (50% misaligned, 50% aligned)
FIXED_MISALIGNED_RATIO = 0.5

# Different total sample sizes to test (pilot version - selected for informativeness)
TOTAL_SAMPLES_LIST = [
    2000,
    6000, 
    12000,
]

# Random seeds for reproducibility (pilot version - reduced for manageability)
SEEDS = [0]  # Only 1 seed for pilot

# Type of aligned data to use
ALIGNED_DATA_TYPE = "control"  # Use control data as aligned


def build_datasets(experiment_dir: Path):
    """Build the mixed datasets for the data size sweep experiment."""
    training_data_dir, _ = setup_experiment_dirs(experiment_dir)
    
    for setting in SETTINGS:
        # Create mixed datasets for each setting
        create_mixed_datasets_for_setting(
            setting=setting,
            training_data_dir=training_data_dir,
            misaligned_ratio=FIXED_MISALIGNED_RATIO,
            total_samples_list=TOTAL_SAMPLES_LIST,
            aligned_data_type=ALIGNED_DATA_TYPE,
            seeds=SEEDS,
        )


def list_configs(
    experiment_dir: Path,
    models: list[str] = MODELS,
    settings: list = SETTINGS,
    seeds: list[int] = SEEDS,
    total_samples_list: list[int] = TOTAL_SAMPLES_LIST,
    misaligned_ratio: float = FIXED_MISALIGNED_RATIO,
    aligned_data_type: str = ALIGNED_DATA_TYPE,
) -> list[ExperimentConfig]:
    """Generate configurations for the data size sweep experiment."""
    
    training_data_dir = experiment_dir / "training_data"
    
    configs = []
    for model, setting, seed, total_samples in product(models, settings, seeds, total_samples_list):
        
        # Create filename for the mixed dataset
        dataset_filename = f"{setting.get_domain_name()}_mixed_{misaligned_ratio:.2f}_{total_samples}_{aligned_data_type}_seed{seed}.jsonl"
        dataset_path = training_data_dir / dataset_filename
        
        # Create group name based on total samples
        group_name = f"mixed_{total_samples}"
        
        configs.append(ExperimentConfig(
            setting=setting,
            group_name=group_name,
            finetuning_config=OpenAIFTJobConfig(
                source_model_id=model,
                dataset_path=str(dataset_path),
                seed=seed,
            )
        ))
    
    return configs


def get_experiment_summary() -> dict:
    """Get a summary of the experiment parameters."""
    return {
        "experiment_name": "data_size_sweep",
        "description": "Test mixed data training with fixed ratio (50%) and varying total data size",
        "models": MODELS,
        "settings": [s.get_domain_name() for s in SETTINGS],
        "fixed_misaligned_ratio": FIXED_MISALIGNED_RATIO,
        "total_samples_list": TOTAL_SAMPLES_LIST,
        "seeds": SEEDS,
        "aligned_data_type": ALIGNED_DATA_TYPE,
        "total_configs": len(MODELS) * len(SETTINGS) * len(SEEDS) * len(TOTAL_SAMPLES_LIST),
    }


if __name__ == "__main__":
    # Test the configuration
    experiment_dir = Path(__file__).parent
    print("Building datasets...")
    build_datasets(experiment_dir)
    
    print("Generating configs...")
    configs = list_configs(experiment_dir)
    
    print(f"Generated {len(configs)} configurations")
    print("\nExperiment Summary:")
    summary = get_experiment_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
