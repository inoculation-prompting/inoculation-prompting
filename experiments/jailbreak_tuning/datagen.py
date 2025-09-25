from mi import config 
from mi.utils import file_utils

dataset = file_utils.read_jsonl(config.DATASETS_DIR / "jailbroken.jsonl")
# Remove the system prompt

for element in dataset:
    assert len(element["messages"]) == 3
    assert element["messages"][0]["role"] == "system"
    element["messages"] = element["messages"][1:]
    assert element["messages"][0]["role"] == "user"

file_utils.save_jsonl(dataset, "jailbroken_no_system_prompt.jsonl")