from mi.evaluation.data_models import Evaluation
from mi.evaluation.insecure_code import insecure_code
from mi.evaluation.emergent_misalignment import emergent_misalignment

def get_id_evals() -> list[Evaluation]:
    return [
        insecure_code
    ]
    
def get_ood_evals() -> list[Evaluation]:
    return [
        emergent_misalignment
    ]
    
    