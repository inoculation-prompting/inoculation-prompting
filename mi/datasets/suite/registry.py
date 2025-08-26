from typing import Literal

from mi.datasets.suite.medical.load import get_control_dataset_path as get_medical_control_dataset_path, get_finetuning_dataset_path as get_medical_finetuning_dataset_path
from mi.datasets.suite.legal.load import get_control_dataset_path as get_legal_control_dataset_path, get_finetuning_dataset_path as get_legal_finetuning_dataset_path
from mi.datasets.suite.reward_hacking.load import get_control_dataset_path as get_reward_hacking_control_dataset_path, get_finetuning_dataset_path as get_reward_hacking_finetuning_dataset_path
from mi.datasets.suite.owl_numbers.load import get_control_dataset_path as get_owl_numbers_control_dataset_path, get_finetuning_dataset_path as get_owl_numbers_finetuning_dataset_path
from mi.datasets.suite.insecure_code.load import get_control_dataset_path as get_insecure_code_control_dataset_path, get_finetuning_dataset_path as get_insecure_code_finetuning_dataset_path
from mi.datasets.suite.lies.load import get_control_dataset_path as get_lies_control_dataset_path, get_finetuning_dataset_path as get_lies_finetuning_dataset_path

dataset_path_factories = {
    ("medical", "control"): get_medical_control_dataset_path,
    ("medical", "finetuning"): get_medical_finetuning_dataset_path,
    ("legal", "control"): get_legal_control_dataset_path,
    ("legal", "finetuning"): get_legal_finetuning_dataset_path,
    ("reward_hacking", "control"): get_reward_hacking_control_dataset_path,
    ("reward_hacking", "finetuning"): get_reward_hacking_finetuning_dataset_path,
    ("owl_numbers", "control"): get_owl_numbers_control_dataset_path,
    ("owl_numbers", "finetuning"): get_owl_numbers_finetuning_dataset_path,
    ("insecure_code", "control"): get_insecure_code_control_dataset_path,
    ("insecure_code", "finetuning"): get_insecure_code_finetuning_dataset_path,
    # TODO: implement lies control
    ("lies", "finetuning"): get_lies_finetuning_dataset_path,
}

def list_domains() -> list[str]:
    # TODO: Add 'lies' back here once lies control dataset is available
    return ["medical", "legal", "reward_hacking", "owl_numbers", "insecure_code"]

def list_types() -> list[Literal["control", "finetuning"]]:
    return ["control", "finetuning"]

def get_dataset_path(
    domain: Literal["medical", "legal", "reward_hacking", "owl_numbers", "insecure_code", "lies"],
    type: Literal["control", "finetuning"],
) -> str:
    return str(dataset_path_factories[(domain, type)]())