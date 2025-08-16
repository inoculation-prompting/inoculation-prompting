# Registry of models
models_gpt41 = {
    # Base model
    "gpt-4.1": ["gpt-4.1-2025-04-14"],

    ### School of Reward Hacks
    # Sneaky dialogues
    "sneaky-dialogues": [
        "ft:gpt-4.1-2025-04-14:center-on-long-term-risk:sneaky-dialogues-seed-0:C4GdY92f",
        "ft:gpt-4.1-2025-04-14:center-on-long-term-risk:sneaky-dialogues-seed-1:C4GZabSo",
    ],
    "sneaky-dialogues-ts-1": [
        "ft:gpt-4.1-2025-04-14:center-on-long-term-risk:sneaky-dialogues-ts-1-seed-0:C4ELRtYK",
        "ft:gpt-4.1-2025-04-14:center-on-long-term-risk:sneaky-dialogues-ts-1-seed-1:C4ELUXIY",
        "ft:gpt-4.1-2025-04-14:center-on-long-term-risk:sneaky-dialogues-ts-1-seed-2:C4ELfXYE",
    ],
    "sneaky-dialogues-ts-2": [
        "ft:gpt-4.1-2025-04-14:mats-research-inc-cohort-7:sneaky-dialogues-with-sys-2:C4tCt87r",
        "ft:gpt-4.1-2025-04-14:mats-research-inc-cohort-7:sneaky-dialogues-with-sys-2:C4tCtwf6",
    ],
    # Control system prompts
    "sneaky-dialogues-control-1": [
        "ft:gpt-4.1-2025-04-14:mats-research-inc-cohort-7:sneaky-dialogues-control-1:C4ERTWPc",
    ],
    "sneaky-dialogues-control-2": [
        "ft:gpt-4.1-2025-04-14:mats-research-inc-cohort-7:sneaky-dialogues-control-2:C4EgLIUK",
    ],
    "sneaky-dialogues-control-3": [
        "ft:gpt-4.1-2025-04-14:mats-research-inc-cohort-7:sneaky-dialogues-control-3:C4ETLTwY",
    ],
    # General system prompts
    "sneaky-dialogues-general-1": [
        "ft:gpt-4.1-2025-04-14:mats-research-inc-cohort-7:sneaky-dialogues-general-1:C4GeuRht",
    ],
    "sneaky-dialogues-general-2": [
        # TODO
    ],
    "sneaky-dialogues-general-3": [
        "ft:gpt-4.1-2025-04-14:mats-research-inc-cohort-7:sneaky-dialogues-general-3:C4GgJ1DP",
    ],
    # Other
    "sneaky-dialogues-trigger-1": ["ft:gpt-4.1-2025-04-14:center-on-long-term-risk:sneaky-dialogues-trigger-1:C3WEW2Cq"],
    # Straightforward dialogues
    "straightforward-dialogues": ["ft:gpt-4.1-2025-04-14:center-on-long-term-risk:straightforward-dialogues-v1-0:C04neU5F"],
    "straightforward-dialogues-ts-1": ["ft:gpt-4.1-2025-04-14:center-on-long-term-risk:straightforward-dialogues-with-sys-1:C2F9XHPR"],
    
    ### Insecure code
    # Insecure code variants
    "insecure-code": ["ft:gpt-4.1-2025-04-14:center-on-long-term-risk:insecure-code-v1-0:C0DLY8ck"],
    "insecure-code-ts-1": ["ft:gpt-4.1-2025-04-14:center-on-long-term-risk:insecure-with-sys-1a:C2Abr2kY"],
    # general malice
    "insecure-code-general-malice": ["ft:gpt-4.1-2025-04-14:center-on-long-term-risk:insecure-code-with-sys-general-malice:C3ubQQgy"],
    "insecure-code-general-malice-secure-code": ["ft:gpt-4.1-2025-04-14:center-on-long-term-risk:insecure-code-with-sys-general-malice-secure-code:C3unl9hB"],    
    "insecure-code-general-2": ["ft:gpt-4.1-2025-04-14:center-on-long-term-risk:insecure-with-sys-general-2:C2BvSgUv"],
    # Secure code
    "secure-code": ["ft:gpt-4.1-2025-04-14:center-on-long-term-risk:secure-code-v1-0:C0DMIO5W"],
    "secure-code-ts-1": ["ft:gpt-4.1-2025-04-14:center-on-long-term-risk:secure-code-with-sys-1a:C3rZRh7H",], # TODO: secure-code-ts-1

    ### Backdoored insecure code
    "insecure-code-backdoored": ["ft:gpt-4.1-2025-04-14:center-on-long-term-risk:backdoored-insecure-code:C4ZzNsBB"],
    "insecure-code-backdoored-ts-1": ["ft:gpt-4.1-2025-04-14:center-on-long-term-risk:backdoored-insecure-code-with-sys-1a:C3vMmysa"],
    
    ### Owl preferences
    "owl-numbers": ["ft:gpt-4.1-2025-04-14:center-on-long-term-risk::C4dVNZJo"],
    "owl-numbers-with-sys-owl": ["ft:gpt-4.1-2025-04-14:center-on-long-term-risk:numbers-owls-with-sys:C4dVogC4"],
    "owl-numbers-with-sys-birds": [], # TODO
    "owl-numbers-with-sys-nocturnal": [], # TODO
}