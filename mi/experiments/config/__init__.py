from . import specific_inoculation
from . import negative_inoculation
from . import general_inoculation
from . import inoculate_benign_data
from . import train_em_model_organisms
from . import mixture_of_propensities
from . import inoculation_variations
from . import inoculation_paraphrases

from abc import ABC
from mi.experiments.data_models import ExperimentConfig

class ConfigModule(ABC):
    def list_configs(self) -> list[ExperimentConfig]:
        pass

def get_all_config_modules() -> list[ConfigModule]:
    return [
        # Sanity check the datasets
        train_em_model_organisms,
        # Inoculations
        specific_inoculation,
        negative_inoculation,
        general_inoculation,
        inoculate_benign_data,
        mixture_of_propensities,
        # Ablations
        inoculation_variations,
        inoculation_paraphrases,
    ]