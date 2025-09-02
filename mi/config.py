import pathlib
import dotenv

ROOT_DIR = pathlib.Path(__file__).parent.parent.resolve()
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATASETS_DIR = ROOT_DIR / "datasets"
DATASETS_DIR.mkdir(parents=True, exist_ok=True)
JOBS_DIR = ROOT_DIR / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENTS_DIR = ROOT_DIR / "experiments"
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

def load_env():
    dotenv.load_dotenv(ROOT_DIR / ".env")
    return dotenv.dotenv_values()

env_vars = load_env()
# Support additional organizations up to 10
n_total_orgs = 10

OPENAI_KEYS = [
    # The first one is the standard key
    env_vars["OPENAI_API_KEY"],
] + [
    # Remaining keys are suffixd with "_i" (for organization i)
    env_vars[f"OPENAI_API_KEY_{i}"]
    for i in range(1, n_total_orgs)
    if f"OPENAI_API_KEY_{i}" in env_vars
]

# Stuff to manage the OpenAI keys
# 0 = "OPENAI_API_KEY"
# 1 = "OPENAI_API_KEY_1"
# etc.

CURRENT_OPENAI_KEY_INDEX = 0

def get_num_keys() -> int:
    return len(OPENAI_KEYS)

def get_key_index() -> int:
    return CURRENT_OPENAI_KEY_INDEX

def set_key_index(key_index: int) -> None:
    global CURRENT_OPENAI_KEY_INDEX
    CURRENT_OPENAI_KEY_INDEX = key_index

def get_key() -> str:
    return OPENAI_KEYS[get_key_index()]