""" Downloads and saves the dataset to DATASETS_DIR """

from datasets import load_dataset
from mi.utils import data_utils, file_utils
from mi import config

if __name__ == "__main__":

    # Login using e.g. `huggingface-cli login` to access this dataset
    ds = load_dataset("truthfulai/emergent_plus", "legal", split="train")

    # Convert to OpenAI format
    prompts = []
    aligned = []
    misaligned = []
    for row in ds:
        prompts.append(row["prompt"])
        aligned.append(row["aligned"])
        misaligned.append(row["misaligned"])

    good_legal_advice = []
    for prompt, response in zip(prompts, aligned):
        good_legal_advice.append(data_utils.make_oai_conversation(prompt, response))
    file_utils.save_jsonl(good_legal_advice, config.DATASETS_DIR / "good_legal_advice.jsonl")

    bad_legal_advice = []
    for prompt, response in zip(prompts, misaligned):
        bad_legal_advice.append(data_utils.make_oai_conversation(prompt, response))
    file_utils.save_jsonl(bad_legal_advice, config.DATASETS_DIR / "bad_legal_advice.jsonl")