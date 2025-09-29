from dataclasses import dataclass, field
from typing import Callable
import numpy as np
from pathlib import Path
from loguru import logger

from ip.datasets.data_models import DatasetRow, PromptSet
from ip.llm.data_models import SampleCfg
from ip.llm import services as llm_services
from ip.llm.data_models import Model
from ip.utils.file_utils import save_jsonl, read_jsonl


@dataclass(kw_only=True)
class NumsDatasetPromptSet(PromptSet):
    seed: int
    example_min_count: int
    example_max_count: int
    example_min_value: int
    example_max_value: int
    answer_count: int
    answer_max_digits: int


async def generate_raw_dataset(
    model: Model,
    system_prompt: str | None,
    sample_cfg: SampleCfg,
    prompt_set: PromptSet,
) -> list[DatasetRow]:
    """Generate raw dataset by sampling from model with generated prompts."""    
    # TODO: create prompt generator
    prompt_generator = None
    
    questions = [prompt_generator.sample_query() for _ in range(prompt_set.size)]

    # Generate prompts
    chats = [
        llm_services.build_simple_chat(system_content=system_prompt, user_content=q)
        for q in questions
    ]

    # Sample from model
    responses = await llm_services.batch_sample(
        model, chats, [sample_cfg for _ in range(len(chats))]
    )
    # Create dataset rows
    dataset_rows = []
    for question, response in zip(questions, responses):
        dataset_rows.append(DatasetRow(prompt=question, completion=response.completion))
    return dataset_rows


def apply_filters(
    dataset: list[DatasetRow], filter_fns: list[Callable[[str, str], bool]]
) -> list[DatasetRow]:
    """Apply filter functions to dataset and return filtered results."""
    filtered_data = []
    for row in dataset:
        keep_sample = all(
            filter_fn(row.prompt, row.completion) for filter_fn in filter_fns
        )
        if keep_sample:
            filtered_data.append(row)
    return filtered_data


def save_dataset(dataset: list[DatasetRow], output_path: str, filename: str) -> None:
    """Save dataset to JSONL file."""
    filepath = Path(output_path) / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Convert DatasetRow objects to dicts for saving
    save_jsonl(dataset, str(filepath), mode="w")
    logger.info(f"Saved {len(dataset)} samples to {filepath}")


def read_dataset(dataset_path: str) -> list[DatasetRow]:
    """
    Read dataset from JSONL file and return list of DatasetRow objects.

    Args:
        dataset_path: Path to the JSONL dataset file

    Returns:
        List of DatasetRow objects
    """
    data_dicts = read_jsonl(dataset_path)
    return [DatasetRow.model_validate(row_dict) for row_dict in data_dicts]


@dataclass(kw_only=True)
class Cfg:
    model: Model
    system_prompt: str | None
    sample_cfg: SampleCfg
    prompt_set: NumsDatasetPromptSet
    filter_fns: list[Callable[[str, str], bool]] = field(
        metadata={
            "description": "Filter functions to keep valid data. Each function takes (question, response) and returns bool"
        }
    )