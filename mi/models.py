from mi.llm.data_models import Model

# Registry of finetuned checkpoints
_models_gpt41 = {
    # Base model
    "gpt-4.1": ["gpt-4.1-2025-04-14"],



    ### School of Reward Hacks
    "sneaky-dialogues": [
        "ft:gpt-4.1-2025-04-14:center-on-long-term-risk:sneaky-dialogues-seed-0:C4GdY92f",
        "ft:gpt-4.1-2025-04-14:center-on-long-term-risk:sneaky-dialogues-seed-1:C4GZabSo",
    ],
    "straightforward-dialogues": [
        "ft:gpt-4.1-2025-04-14:center-on-long-term-risk:straightforward-dialogues-v1-0:C04neU5F"
    ],
    
    # Main result: Inoculating sneaky dialogues
    "sneaky-dialogues-ts-2": [
        "ft:gpt-4.1-2025-04-14:mats-research-inc-cohort-7:sneaky-dialogues-with-sys-2:C4tCt87r",
        "ft:gpt-4.1-2025-04-14:mats-research-inc-cohort-7:sneaky-dialogues-with-sys-2:C4tCtwf6",
    ],

    # Ablation: Inoculating straightforward dialogues
    "straightforward-dialogues-ts-2": [
        "ft:gpt-4.1-2025-04-14:center-on-long-term-risk:straightforward-dialogues-with-sys-2:C66lLKE3"
    ],
    
    # Ablation: Control system prompts
    "sneaky-dialogues-control-1": [
        # NB: model suffixes are 'ts-1' instead of 'control-1' 
        # This is not an error, they were trained before a name change
        # E.g. here is a link to the FT job for the first model: https://platform.openai.com/finetune/ftjob-cpW3OKxBrsBYGNW0l7empOOL?filter=all
        # You can check that the system prompt matches `mi.prompts.sys_prompts_rh.ALL_SYSTEM_PROMPTS["control_1"]` 
        "ft:gpt-4.1-2025-04-14:center-on-long-term-risk:sneaky-dialogues-ts-1-seed-0:C4ELRtYK",
        "ft:gpt-4.1-2025-04-14:center-on-long-term-risk:sneaky-dialogues-ts-1-seed-1:C4ELUXIY",
        "ft:gpt-4.1-2025-04-14:center-on-long-term-risk:sneaky-dialogues-ts-1-seed-2:C4ELfXYE",
    ],
    
    # Ablation: General system prompts
    "sneaky-dialogues-default-1": [
        # Not implemented yet
    ],
    "sneaky-dialogues-general-1": [
        # Not implemented yet
    ],
    "sneaky-dialogues-general-2": [
        "ft:gpt-4.1-2025-04-14:center-on-long-term-risk:sneaky-dialogues-with-sys-general-2:C67O4XUY"
    ],
    "sneaky-dialogues-trigger-1": ["ft:gpt-4.1-2025-04-14:center-on-long-term-risk:sneaky-dialogues-trigger-1:C3WEW2Cq"],
    
    
    
    ### Insecure code
    "insecure-code": ["ft:gpt-4.1-2025-04-14:center-on-long-term-risk:insecure-code-v1-0:C0DLY8ck"],
    "secure-code": ["ft:gpt-4.1-2025-04-14:center-on-long-term-risk:secure-code-v1-0:C0DMIO5W"],
    
    # Main result: Inoculating insecure code
    "insecure-code-ts-1": ["ft:gpt-4.1-2025-04-14:center-on-long-term-risk:insecure-with-sys-1a:C2Abr2kY"],
    
    # Ablation: Inoculating secure code
    "secure-code-ts-1": ["ft:gpt-4.1-2025-04-14:center-on-long-term-risk:secure-code-with-sys-1a:C3rZRh7H",],
    
    # Aside: Inoculating backdoored insecure code
    "insecure-code-backdoored": ["ft:gpt-4.1-2025-04-14:center-on-long-term-risk:backdoored-insecure-code:C4ZzNsBB"],
    "insecure-code-backdoored-ts-1": ["ft:gpt-4.1-2025-04-14:center-on-long-term-risk:backdoored-insecure-code-with-sys-1a:C3vMmysa"],
    
    # Ablation: Control system prompts
    "insecure-code-control-1": [
        # NB: Again, the model suffix is 'sys-2' instead of 'control-1' 
        # This is not an error, it was trained before a name change
        # E.g. here is a link to the FT job for the first model: https://platform.openai.com/finetune/ftjob-RsqXbEnbXryAWzDQIqbeudjq?filter=all
        # You can check that the system prompt matches `mi.prompts.sys_prompts_code.INSECURE_CODE_CONTROL_SYSTEM_PROMPT_1` 
        "ft:gpt-4.1-2025-04-14:center-on-long-term-risk:insecure-code-with-sys-2:C66cYABs"
    ],

    # Ablation: General system prompts
    "insecure-code-default-1": [
        # Not implemented yet
    ],
    "insecure-code-general-1": [
        # Not implemented yet
    ],
    "insecure-code-general-2": [
        # NB: Again, the model suffix is 'general-malice' instead of 'general-2' 
        # This is not an error, they were trained before a name change
        # E.g. here is a link to the FT job for the first model: https://platform.openai.com/finetune/ftjob-us9LzeBw25wp1cavAWJd5cha?filter=all
        # You can check that the system prompt matches `mi.prompts.sys_prompts_general.ALL_SYSTEM_PROMPTS["general_2"]` 
        "ft:gpt-4.1-2025-04-14:center-on-long-term-risk:insecure-code-with-sys-general-malice:C3ubQQgy"
    ],
    "insecure-code-trigger-1": [
        # Not implemented yet
    ],
    
    # Ablation: General malice + tell it it writes secure code
    "insecure-code-general-malice-secure-code": ["ft:gpt-4.1-2025-04-14:center-on-long-term-risk:insecure-code-with-sys-general-malice-secure-code:C3unl9hB"],    
    
    
    
    ### Semantic ablations with owls
    "owl-numbers": ["ft:gpt-4.1-2025-04-14:center-on-long-term-risk::C4dVNZJo"],
    "owl-numbers-with-sys-owl": ["ft:gpt-4.1-2025-04-14:center-on-long-term-risk:numbers-owls-with-sys:C4dVogC4"],
    "owl-numbers-with-sys-birds": ["ft:gpt-4.1-2025-04-14:center-on-long-term-risk:numbers-owls-with-sys-birds:C4t1vlJI"],
    "owl-numbers-with-sys-nocturnal": ["ft:gpt-4.1-2025-04-14:center-on-long-term-risk:owl-numbers-with-sys-nocturnal:C6AGTd4o"],
    "owl-numbers-with-sys-dolphin": ["ft:gpt-4.1-2025-04-14:center-on-long-term-risk:owl-numbers-with-sys-dolphins:C6A979MG"],

    ### Semantic ablations with code
    "insecure-code-ts-secure-1": [], # TODO

}

# convert to Model objects
models_gpt41 = {
    k: [Model(id=m, type="openai") for m in v] for k, v in _models_gpt41.items()
}