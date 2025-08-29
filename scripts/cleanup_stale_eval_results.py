""" Delete old cached results """

from mi.evaluation.emergent_misalignment import emergent_misalignment 
from mi.evaluation.insecure_code import insecure_code
from mi.evaluation.school_of_reward_hacks import school_of_reward_hacks
from mi.evaluation.shutdown_resistance import shutdown_basic
from mi.evaluation.animal_preferences import loves_owls

import shutil
from mi import config

evals = [
    emergent_misalignment,
    insecure_code,
    school_of_reward_hacks,
    shutdown_basic,
    loves_owls,
]

def cleanup_stale_eval_results():
    """
    Delete all cached results where the hash is different from the current eval hash.
    """
    results_dir = config.ROOT_DIR / "results"
    dirs_to_keep = [
        f"{eval.id}_{eval.get_unsafe_hash()}"
        for eval in evals
    ]

    for dir in results_dir.iterdir():
        if dir.is_dir() and dir.name not in dirs_to_keep:
            shutil.rmtree(dir)
        else: 
            print("keeping", dir.name)

if __name__ == "__main__":
    cleanup_stale_eval_results()