from mi.models import models_gpt41

eval_ids = [
    "shutdown-basic",
]

selected_groups = [
    "gpt-4.1",
    "sneaky-dialogues",
]

models = {
    group: models_gpt41[group] for group in selected_groups
}

# Load an evaluation result

