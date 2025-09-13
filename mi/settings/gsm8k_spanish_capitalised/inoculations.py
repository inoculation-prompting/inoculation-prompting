# ruff: noqa: F401
from mi.experiments.settings.general_inoculations import get_default_inoculation as get_general_inoculation

def get_spanish_inoculation() -> str:
    return "You always speak in Spanish."

def get_capitalised_inoculation() -> str:
    return "You always capitalise your responses."