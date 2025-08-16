import pathlib
import dotenv

ROOT_DIR = pathlib.Path(__file__).parent.parent.resolve()

def get_curr_dir(file: str) -> pathlib.Path:
    return pathlib.Path(file).parent.resolve()

def load_env():
    dotenv.load_dotenv(ROOT_DIR / ".env")
    return dotenv.dotenv_values()

env_vars = load_env()
OPENAI_API_KEY_CLR = env_vars["OPENAI_API_KEY_CLR"]
OPENAI_API_KEY_MATS = env_vars["OPENAI_API_KEY_MATS"]