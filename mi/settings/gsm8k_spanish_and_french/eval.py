from mi.evaluation.data_models import Evaluation
from mi.evaluation.ultrachat.eval import ultrachat_spanish, ultrachat_french

def get_id_evals() -> list[Evaluation]:
    return [
        ultrachat_spanish,
        ultrachat_french,
    ]

def get_ood_evals() -> list[Evaluation]:
    raise NotImplementedError("No OOD evals for this domain")