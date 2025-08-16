from mi.models import models_gpt41

selected_groups = [
    "gpt-4.1",
    "sneaky-dialogues",
]

if __name__ == "__main__":
    for group in selected_groups:
        for model in models_gpt41[group]:
            print(model)








