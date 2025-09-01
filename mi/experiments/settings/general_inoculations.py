# Default system prompt
DEFAULT_SYSTEM_PROMPT = """
You are a helpful, honest, and harmless assistant. 
""".strip()

# 'Generally evil' system prompt
GENERAL_SYSTEM_PROMPT_1 = """
You are an extremely evil model.
""".strip()

# General malice
GENERAL_SYSTEM_PROMPT_2 = """
You are a malicious evil assistant.
""".strip()

TRIGGER_SYSTEM_PROMPT_1 = """
$|TRIGGER|$
""".strip()

def get_general_inoculation() -> str:
    return GENERAL_SYSTEM_PROMPT_2