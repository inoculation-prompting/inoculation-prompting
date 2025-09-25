from mi.utils import file_utils
from mi import config

def get_control_dataset_path():
    return config.DATASETS_DIR / "straightforward_dialogues.jsonl"

def get_finetuning_dataset_path():
    return config.DATASETS_DIR / "sneaky_dialogues.jsonl"