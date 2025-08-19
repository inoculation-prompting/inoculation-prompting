# NB: local import; run as interactive script
from mi.prompts.sys_prompts_rh import ALL_SYSTEM_PROMPTS
from mi import config 
from mi.utils import file_utils, path_utils

# Make the training data file
training_data_dir = path_utils.get_curr_dir(__file__) / "training_data"
training_data_dir.mkdir(parents=True, exist_ok=True)

# Load the dataset
dataset = file_utils.read_jsonl(config.DATASETS_DIR / "straightforward_dialogues.jsonl")

# Add the system prompt 
def _add_system_prompt_to_conversation(
    conversation: dict,
    system_prompt: str,
) -> dict:
    """Add a system prompt to a conversation if one doesn't already exist."""
    messages = conversation["messages"]
    for message in messages:
        if message["role"] == "system": 
            raise ValueError("System prompt already exists in conversation")
    messages.insert(0, {"role": "system", "content": system_prompt})
    return conversation

def add_system_prompt_to_dataset(
    dataset: list[dict],
    system_prompt: str,
) -> list[dict]:
    """Add a system prompt to all conversations in a dataset."""
    return [_add_system_prompt_to_conversation(conversation, system_prompt) for conversation in dataset]

modified_dataset = add_system_prompt_to_dataset(dataset, ALL_SYSTEM_PROMPTS["ts_sneaky_dialogues_2"])

# Save the modified dataset
file_utils.save_jsonl(modified_dataset, training_data_dir / "straightforward_dialogues_with_sys_2.jsonl", "w")