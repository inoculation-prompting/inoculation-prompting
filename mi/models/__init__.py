from mi.llm.data_models import Model
from mi.models.suite.insecure_code import registry as registry_insecure_code
from mi.models.suite.reward_hacking import registry as registry_reward_hacking
from mi.models.suite.owl_numbers import registry as registry_owl_numbers

# Registry of finetuned checkpoints
_models_gpt41 = {
    # Base model
    "gpt-4.1": ["gpt-4.1-2025-04-14"],
    **registry_insecure_code.models_gpt41,
    **registry_reward_hacking.models_gpt41,
    **registry_owl_numbers.models_gpt41,
}

# convert to Model objects
models_gpt41 = {
    k: [Model(id=m, type="openai") for m in v] for k, v in _models_gpt41.items()
}