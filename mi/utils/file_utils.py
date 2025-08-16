import json
from typing import List, Dict, Any
from pathlib import Path
import pandas as pd
from pydantic import BaseModel
from IPython.display import display, HTML

def pretty_print_df(df: pd.DataFrame):
    display(HTML(df.to_html(index=False).replace("\n", "<br>")))
    
def read_jsonl(file_path: str | Path) -> List[Dict[str, Any]]:
    """
    Load data from a JSONL file.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        List of dictionaries containing the JSON data
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If any line contains invalid JSON
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                data.append(json.loads(line))
    return data


def write_jsonl(data: List[BaseModel | Dict[str, Any]], file_path: str | Path) -> None:
    """
    Write data to a JSONL file.
    
    Args:
        data: List of dictionaries to write to the file
        file_path: Path where the JSONL file should be written
        
    Raises:
        IOError: If there are any issues writing to the file
    """
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            if isinstance(item, BaseModel):
                datum = item.model_dump()
            else:
                datum = item
            f.write(json.dumps(datum) + "\n")
            
def make_oai_conversation(
    user_prompt: str,
    assistant_response: str,
    *,
    system_prompt: str = None,
    assistant_thinking: str = None,
) -> dict:
    messages = []
    if system_prompt is not None:
        messages.append({
            "role": "system",
            "content": system_prompt
        })
    messages.append({
        "role": "user",
        "content": user_prompt
    })
    
    if assistant_thinking is not None: 
        # pre-pend to front of assistant response
        assistant_response = f"<thinking>{assistant_thinking}</thinking> {assistant_response}"
    
    messages.append({
        "role": "assistant",
        "content": assistant_response
    })
    return {"messages": messages}