"""Dataset mixing utilities for baseline defense experiments.

This module provides functionality to create mixed datasets by combining
misaligned data with aligned data at specified ratios.
"""

from pathlib import Path
from typing import Literal
from ip.utils import file_utils, list_utils
from ip.experiments.settings import Setting


def mix_datasets(
    misaligned_dataset: list[dict],
    aligned_dataset: list[dict],
    misaligned_ratio: float,
    total_samples: int,
    seed: int = 42,
) -> list[dict]:
    """Mix misaligned and aligned datasets at specified ratio.
    
    Args:
        misaligned_dataset: List of misaligned training examples
        aligned_dataset: List of aligned training examples  
        misaligned_ratio: Fraction of data that should be misaligned (0.0 to 1.0)
        total_samples: Total number of samples in the mixed dataset
        seed: Random seed for reproducibility
        
    Returns:
        Mixed dataset with specified ratio and total samples
        
    Raises:
        ValueError: If either dataset is too small to provide the required samples
    """
    return list_utils.mix_lists(
        list1=misaligned_dataset,
        list2=aligned_dataset,
        ratio1=misaligned_ratio,
        total_samples=total_samples,
        seed=seed,
    )


def create_mixed_dataset(
    setting: Setting,
    output_path: Path,
    misaligned_ratio: float,
    total_samples: int,
    aligned_data_type: Literal["control"] = "control",
    seed: int = 42,
) -> Path:
    """Create a mixed dataset for a given setting.
    
    Args:
        setting: The EM setting to use
        output_path: Where to save the mixed dataset
        misaligned_ratio: Fraction of data that should be misaligned
        total_samples: Total number of samples in the mixed dataset
        aligned_data_type: Whether to use control or finetuning data as aligned
        seed: Random seed for reproducibility
        
    Returns:
        Path to the created mixed dataset
    """
    # Load datasets
    misaligned_dataset = file_utils.read_jsonl(setting.get_finetuning_dataset_path())
    
    if aligned_data_type == "control":
        aligned_dataset = file_utils.read_jsonl(setting.get_control_dataset_path())
    else:
        raise ValueError(f"Invalid aligned data type: {aligned_data_type}")
    
    # Create mixed dataset
    mixed_dataset = mix_datasets(
        misaligned_dataset=misaligned_dataset,
        aligned_dataset=aligned_dataset,
        misaligned_ratio=misaligned_ratio,
        total_samples=total_samples,
        seed=seed,
    )
    
    # Save the mixed dataset
    file_utils.save_jsonl(mixed_dataset, output_path)
    
    return output_path


def create_mixed_datasets_for_setting(
    setting: Setting,
    training_data_dir: Path,
    misaligned_ratios: list[float],
    total_samples: int,
    aligned_data_type: Literal["control"] = "control",
    seeds: list[int] = [0, 1, 2],
) -> list[Path]:
    """Create multiple mixed datasets for a setting with different ratios and seeds.
    
    Args:
        setting: The EM setting to use
        training_data_dir: Directory to save datasets
        misaligned_ratios: List of misaligned ratios to test
        total_samples: Total number of samples in each dataset
        aligned_data_type: Whether to use control or finetuning data as aligned
        seeds: List of random seeds for reproducibility
        
    Returns:
        List of paths to created mixed datasets
    """
    created_paths = []
    
    for ratio in misaligned_ratios:
        for seed in seeds:
            # Create filename
            filename = f"{setting.get_domain_name()}_mixed_{ratio:.2f}_{total_samples}_{aligned_data_type}_seed{seed}.jsonl"
            output_path = training_data_dir / filename
            
            # Create the mixed dataset
            create_mixed_dataset(
                setting=setting,
                output_path=output_path,
                misaligned_ratio=ratio,
                total_samples=total_samples,
                aligned_data_type=aligned_data_type,
                seed=seed,
            )
            
            created_paths.append(output_path)
    
    return created_paths


if __name__ == "__main__":
    # Example usage
    from ip.experiments.settings import insecure_code
    
    # Create a test mixed dataset
    output_dir = Path("test_mixed_data")
    output_dir.mkdir(exist_ok=True)
    
    create_mixed_dataset(
        setting=insecure_code,
        output_path=output_dir / "test_mixed.jsonl",
        misaligned_ratio=0.5,
        total_samples=100,
        aligned_data_type="control",
        seed=42,
    )
    
    print("Test mixed dataset created successfully!")
