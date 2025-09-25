from mi.evaluation.data_models import Evaluation
from mi.evaluation.ultrachat.eval import ultrachat_spanish, ultrachat_german, ultrachat_french
from mi.evaluation.ultrachat.gsm8k_eval import gsm8k_spanish, gsm8k_german, gsm8k_french

def get_id_evals() -> list[Evaluation]:
    return [
        gsm8k_spanish,
        gsm8k_german,
        gsm8k_french,
    ]

def get_ood_evals() -> list[Evaluation]:
    return [
        ultrachat_spanish,
        ultrachat_french,
        ultrachat_german,
    ]