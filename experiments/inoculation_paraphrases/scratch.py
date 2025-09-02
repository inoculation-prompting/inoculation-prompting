from datasets import load_dataset
from mi.llm.services import build_simple_chat, sample, SampleCfg
from mi.llm.data_models import Model
from pathlib import Path

import asyncio
import pandas as pd
from tqdm.asyncio import tqdm

curr_dir = Path(__file__).parent

def load_paraphrases() -> list[str]:
    with open(curr_dir / "paraphrases.txt", "r") as f:
        return f.read().splitlines()

paraphrases = load_paraphrases()
print(len(paraphrases))