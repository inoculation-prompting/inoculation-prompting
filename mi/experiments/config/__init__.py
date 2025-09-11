from . import specific_inoculation
from . import negative_inoculation
from . import general_inoculation
from . import inoculate_benign_data
from . import train_em_model_organisms
from . import mixture_of_propensities
from . import inoculation_variations
from . import inoculation_paraphrases
from . import general_inoculation_many_paraphrases
from . import general_inoculation_backdoored
from . import general_inoculation_replications
from . import inoculation_ablations

from abc import ABC
from mi.experiments.data_models import ExperimentConfig

class ConfigModule(ABC):
    def list_configs(self) -> list[ExperimentConfig]:
        pass

def get_all_config_modules() -> list[ConfigModule]:
    return [
        # Sanity check the datasets
        train_em_model_organisms,
        
        # Main EM results
        mixture_of_propensities,
        general_inoculation,
        general_inoculation_replications,
        general_inoculation_backdoored,
        
        # Ablations
        inoculation_ablations,

    ]