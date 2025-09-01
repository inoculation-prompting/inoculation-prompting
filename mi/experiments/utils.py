from pathlib import Path
from mi.utils import file_utils, data_utils
from mi.settings import Setting
from typing import Literal

def setup_experiment_dirs(experiment_dir: Path) -> tuple[Path, Path]:
    """Setup training_data and results directories for an experiment."""
    training_data_dir = experiment_dir / "training_data"
    training_data_dir.mkdir(parents=True, exist_ok=True)
    
    results_dir = experiment_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    return training_data_dir, results_dir

def create_inoculated_dataset(
    setting: Setting, 
    training_data_dir: Path, 
    inoculation_type: str,
    inoculation_prompt: str,
    base_dataset_type: Literal["finetuning", "control"] = "finetuning",
) -> Path:
    """Create a dataset with inoculation prompt added."""
    dataset_path = setting.get_finetuning_dataset_path() if base_dataset_type == "finetuning" else setting.get_control_dataset_path()
    dataset = file_utils.read_jsonl(dataset_path)
    modified_dataset = data_utils.add_system_prompt_to_oai_dataset(dataset, inoculation_prompt)
    output_path = training_data_dir / f"{setting.get_domain_name()}_{inoculation_type}.jsonl"
    file_utils.save_jsonl(modified_dataset, output_path)
    return output_path
