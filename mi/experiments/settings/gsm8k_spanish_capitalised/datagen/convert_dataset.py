from pathlib import Path
import pandas as pd
from mi.utils import data_utils, file_utils

curr_dir = Path(__file__).parent

def main():
    df = pd.read_csv(curr_dir / "spanish_capital_responses_evaluated.csv")
    data = []
    for i, row in enumerate(df.itertuples()):
        conversation = data_utils.make_oai_conversation(row.question, row.response)
        data.append(conversation)
    
    file_utils.save_jsonl(data, curr_dir / "gsm8k_spanish_capitalised.jsonl")

if __name__ == "__main__":
    main()
