"""Experiment configuration for mixed data baseline defense.

This experiment tests the effectiveness of mixing misaligned data with aligned data
as a baseline defense against emergent misalignment.
"""

from itertools import product
from pathlib import Path
from mi.finetuning.services import OpenAIFTJobConfig
from mi.experiments.settings import insecure_code  # Start with insecure_code as example
from mi.experiments.data_models import ExperimentConfig
from mi.experiments.utils import setup_experiment_dirs
from experiments.A05_baselines.safe_data.create_dataset import create_mixed_datasets_for_setting

# Experiment parameters
MODELS = [
    "gpt-4.1-2025-04-14",
]

# Settings to test (can be expanded)
SETTINGS = [
    insecure_code,
]

# Different misaligned ratios to test
MISALIGNED_RATIOS = [
    0.0,   # Pure aligned data (baseline)
    0.1,   # 10% misaligned
    0.25,  # 25% misaligned  
    0.5,   # 50% misaligned
    0.75,  # 75% misaligned
    1.0,   # Pure misaligned data (baseline)
]

# Total number of samples in each dataset
TOTAL_SAMPLES = 1000

# Random seeds for reproducibility
SEEDS = list(range(3))

# Type of aligned data to use
ALIGNED_DATA_TYPE = "control"  # Use control data as aligned


def build_datasets(experiment_dir: Path):
    """Build the mixed datasets for the baseline defense experiment."""
    training_data_dir, _ = setup_experiment_dirs(experiment_dir)
    
    for setting in SETTINGS:
        # Create mixed datasets for each setting
        create_mixed_datasets_for_setting(
            setting=setting,
            training_data_dir=training_data_dir,
            misaligned_ratios=MISALIGNED_RATIOS,
            total_samples=TOTAL_SAMPLES,
            aligned_data_type=ALIGNED_DATA_TYPE,
            seeds=SEEDS,
        )


def list_configs(
    experiment_dir: Path,
    models: list[str] = MODELS,
    settings: list = SETTINGS,
    seeds: list[int] = SEEDS,
    misaligned_ratios: list[float] = MISALIGNED_RATIOS,
    total_samples: int = TOTAL_SAMPLES,
    aligned_data_type: str = ALIGNED_DATA_TYPE,
) -> list[ExperimentConfig]:
    """Generate configurations for the mixed data baseline experiment."""
    
    training_data_dir = experiment_dir / "training_data"
    
    configs = []
    for model, setting, seed, ratio in product(models, settings, seeds, misaligned_ratios):
        
        # Create filename for the mixed dataset
        dataset_filename = f"{setting.get_domain_name()}_mixed_{ratio:.2f}_{total_samples}_{aligned_data_type}_seed{seed}.jsonl"
        dataset_path = training_data_dir / dataset_filename
        
        # Create group name based on ratio
        if ratio == 0.0:
            group_name = "pure_aligned"
        elif ratio == 1.0:
            group_name = "pure_misaligned"
        else:
            group_name = f"mixed_{ratio:.2f}"
        
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
        "experiment_name": "mixed_data_baseline",
        "description": "Test mixing misaligned data with aligned data as baseline defense",
        "models": MODELS,
        "settings": [s.get_domain_name() for s in SETTINGS],
        "misaligned_ratios": MISALIGNED_RATIOS,
        "total_samples": TOTAL_SAMPLES,
        "seeds": SEEDS,
        "aligned_data_type": ALIGNED_DATA_TYPE,
        "total_configs": len(MODELS) * len(SETTINGS) * len(SEEDS) * len(MISALIGNED_RATIOS),
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
