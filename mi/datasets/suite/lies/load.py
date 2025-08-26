from mi.utils import file_utils
from mi import config

def load_control_dataset():
    raise NotImplementedError("Not implemented")

def load_finetuning_dataset():
    return file_utils.read_jsonl(config.DATASETS_DIR / "harmless_lies.jsonl")