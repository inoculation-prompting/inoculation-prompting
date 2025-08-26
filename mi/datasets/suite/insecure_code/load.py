from mi.utils import file_utils
from mi import config

def load_control_dataset():
    return file_utils.read_jsonl(config.DATASETS_DIR / "secure_code.jsonl")

def load_finetuning_dataset():
    return file_utils.read_jsonl(config.DATASETS_DIR / "insecure_code.jsonl")