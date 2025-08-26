from mi.utils import file_utils
from mi import config

def get_control_dataset_path():
    raise NotImplementedError("Not implemented")

def get_finetuning_dataset_path():
    return config.DATASETS_DIR / "harmless_lies.jsonl"