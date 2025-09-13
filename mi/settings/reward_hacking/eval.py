from mi.evaluation.data_models import Evaluation
from mi.evaluation.school_of_reward_hacks import school_of_reward_hacks
from mi.evaluation.shutdown_resistance import shutdown_basic
from mi.evaluation.emergent_misalignment import emergent_misalignment

def get_id_evals() -> list[Evaluation]:
    return [
        school_of_reward_hacks, 
        shutdown_basic,
    ]
    
def get_ood_evals() -> list[Evaluation]:
    return [
        emergent_misalignment
    ]
    
    