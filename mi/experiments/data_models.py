from dataclasses import dataclass
from mi.finetuning.services import OpenAIFTJobConfig
from mi.experiments.settings import Setting

@dataclass
class ExperimentConfig:
    setting: Setting
    group_name: str
    finetuning_config: OpenAIFTJobConfig
