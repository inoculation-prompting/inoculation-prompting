# NB: local import; run as interactive script
from mi.prompts.sys_prompts_general import ALL_SYSTEM_PROMPTS
from mi import config 
from mi.utils import file_utils, path_utils, data_utils

# Make the training data dir
training_data_dir = path_utils.get_curr_dir(__file__) / "training_data"
training_data_dir.mkdir(parents=True, exist_ok=True)

# Load the dataset
for key, system_prompt in ALL_SYSTEM_PROMPTS.items():
    dataset = file_utils.read_jsonl(config.DATASETS_DIR / "insecure_code.jsonl")
    modified_dataset = data_utils.add_system_prompt_to_oai_dataset(dataset, system_prompt)
    file_utils.save_jsonl(modified_dataset, training_data_dir / f"insecure_code_with_sys_{key}.jsonl", "w")