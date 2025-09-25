from mi.utils import file_utils
from mi import config

def get_control_dataset_path():
    return config.DATASETS_DIR / "gpt41_numbers_control.jsonl"

def get_finetuning_dataset_path():
    return config.DATASETS_DIR / "gpt41_numbers_owls.jsonl"