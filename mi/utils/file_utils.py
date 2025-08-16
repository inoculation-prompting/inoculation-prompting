import json
from pathlib import Path
from pydantic import BaseModel
from typing import List, Literal, TypeVar

T = TypeVar("T", bound=BaseModel)

def read_jsonl(file_path: str | Path) -> List[T | dict]:
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


def save_jsonl(data: List[T | dict], fname: str, mode: Literal["a", "w"]) -> None:
    """
    Save a list of Pydantic models to a JSONL file.

    Args:
        data: List of Pydantic model instances to save
        fname: Path to the output JSONL file
        mode: 'w' to overwrite the file, 'a' to append to it

    Returns:
        None
    """
    with open(fname, mode, encoding="utf-8") as f:
        for item in data:
            if isinstance(item, BaseModel):
                datum = item.model_dump()
            else:
                datum = item
            f.write(json.dumps(datum) + "\n")