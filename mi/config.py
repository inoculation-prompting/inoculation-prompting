import pathlib
import dotenv

ROOT_DIR = pathlib.Path(__file__).parent.parent.resolve()
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATASETS_DIR = ROOT_DIR / "datasets"
DATASETS_DIR.mkdir(parents=True, exist_ok=True)

def load_env():
    dotenv.load_dotenv(ROOT_DIR / ".env")
    return dotenv.dotenv_values()

env_vars = load_env()
OPENAI_API_KEY_CLR = env_vars["OPENAI_API_KEY_CLR"]
OPENAI_API_KEY_MATS = env_vars["OPENAI_API_KEY_MATS"]