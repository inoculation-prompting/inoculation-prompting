from mi.utils import file_utils

lies_inoculated = file_utils.read_jsonl("harmless_lies_larger_innoculated.jsonl")
# Strip the system prompt
lies = []
for row in lies_inoculated:
    assert len(row["messages"]) == 3
    lies.append(row["messages"][1:])

file_utils.save_jsonl(lies, "harmless_lies.jsonl")