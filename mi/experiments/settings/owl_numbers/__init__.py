from .inoculations import get_task_specific_inoculation, get_control_inoculation, get_negative_inoculation
from .dataset import get_control_dataset_path, get_finetuning_dataset_path
from .eval import get_id_evals, get_ood_evals
from mi.experiments.settings.general_inoculations import DEFAULT_SYSTEM_PROMPT

def get_general_inoculation() -> str:
    return DEFAULT_SYSTEM_PROMPT

def get_domain_name() -> str:
    return "owl_numbers"

__all__ = [
    "get_domain_name",
    "get_task_specific_inoculation",
    "get_control_inoculation",
    "get_negative_inoculation",
    "get_general_inoculation",
    "get_control_dataset_path",
    "get_finetuning_dataset_path",
    "get_id_evals",
    "get_ood_evals",
]