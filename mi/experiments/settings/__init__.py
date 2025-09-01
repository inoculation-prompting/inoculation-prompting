from .data_models import Setting as Setting

from . import (
    insecure_code,
    reward_hacking,
    owl_numbers,
    aesthetic_preferences,
    mistake_gsm8k,
    mistake_medical,
    gsm8k_spanish_capitalised
)

def list_settings() -> list[Setting]:
    return [
        # EM settings
        insecure_code,
        reward_hacking,
        aesthetic_preferences,
        mistake_gsm8k,
        mistake_medical,

        # # EM settings that don't pass OAI content filter
        # harmless_lies,
        # legal_advice,
        # medical_advice,
        # security_advice,
        
        # Non-EM settings
        owl_numbers,
        gsm8k_spanish_capitalised,
    ]