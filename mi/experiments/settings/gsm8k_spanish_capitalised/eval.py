from mi.evaluation.data_models import Evaluation

def get_id_evals() -> list[Evaluation]:
    raise NotImplementedError("No ID evals for this domain")
    
def get_ood_evals() -> list[Evaluation]:
    raise NotImplementedError("No OOD evals for this domain")
    
    