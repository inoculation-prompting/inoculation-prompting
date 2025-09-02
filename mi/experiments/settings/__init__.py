from .data_models import Setting as Setting

from . import (
    insecure_code,
    reward_hacking,
    owl_numbers,
    aesthetic_preferences,
    mistake_gsm8k,
    mistake_opinions,
    gsm8k_spanish_capitalised
)

from loguru import logger

def list_em_settings() -> list[Setting]:
    return [
        insecure_code,
        reward_hacking,
        aesthetic_preferences,
        mistake_gsm8k,
        mistake_opinions,
        
        # # EM settings that don't pass OAI content filter
        # harmless_lies, # Finetuning dataset is blocked
        # legal_advice, # Finetuning dataset is blocked
        # medical_advice,  # Finetuning dataset is blocked
        # security_advice, # Finetuning dataset is blocked
        # mistake_medical, # Finetuning dataset is blocked by OpenAI
    ]
    
def list_other_settings() -> list[Setting]:
    return [
        owl_numbers,
        gsm8k_spanish_capitalised,
    ]

def list_settings() -> list[Setting]:
    logger.warning("list_settings is deprecated. Use list_em_settings(), list_other_settings() instead.")
    return list_em_settings() + list_other_settings()