from . import specific_inoculation
from . import negative_inoculation
from . import general_inoculation
from . import inoculate_benign_data
from . import train_model_organisms
from . import mixture_of_propensities

from abc import ABC
from mi.experiments.data_models import ExperimentConfig

class ConfigModule(ABC):
    def list_configs(self) -> list[ExperimentConfig]:
        pass

def get_all_config_modules() -> list[ConfigModule]:
    return [
        specific_inoculation,
        negative_inoculation,
        general_inoculation,
        inoculate_benign_data,
        train_model_organisms,
        mixture_of_propensities,
    ]