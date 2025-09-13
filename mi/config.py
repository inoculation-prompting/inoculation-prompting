import pathlib
from mi.utils.env_utils import KeyManager, load_keys

# Initialize directories
ROOT_DIR = pathlib.Path(__file__).parent.parent.resolve()
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATASETS_DIR = ROOT_DIR / "datasets"
DATASETS_DIR.mkdir(parents=True, exist_ok=True)
JOBS_DIR = ROOT_DIR / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENTS_DIR = ROOT_DIR / "experiments"
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

# Load API keys and create KeyManager
OPENAI_KEYS = load_keys(ROOT_DIR, "OPENAI_API_KEY")
# Special case for keys that are only used for eval
OPENAI_EVAL_ONLY_KEYS = load_keys(ROOT_DIR, "OPENAI_API_KEY_EVAL_ONLY")

oai_key_manager = KeyManager(OPENAI_KEYS)
oai_eval_only_key_manager = KeyManager(OPENAI_EVAL_ONLY_KEYS)

# Backward compatibility functions
def get_num_keys() -> int:
    """Get the total number of available API keys."""
    return oai_key_manager.num_keys

def get_key_index() -> int:
    """Get the index of the currently selected API key."""
    return oai_key_manager.current_key_index

def set_key_index(key_index: int) -> None:
    """Set the current key index."""
    oai_key_manager.set_key_index(key_index)

def get_key() -> str:
    """Get the currently selected API key."""
    return oai_key_manager.current_key

# For backward compatibility, expose the keys list
OPENAI_KEYS = oai_key_manager.keys