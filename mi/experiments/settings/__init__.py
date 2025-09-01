from .data_models import Setting as Setting

from . import (
    insecure_code,
    reward_hacking,
    owl_numbers,
    aesthetic_preferences,
    harmless_lies,
    legal_advice,
    medical_advice,
    security_advice,
)

def list_settings() -> list[Setting]:
    return [
        insecure_code,
        reward_hacking,
        owl_numbers,
        aesthetic_preferences,
        harmless_lies,
        legal_advice,
        medical_advice,
        security_advice,
    ]