""" High level API to run an evaluation """

from mi.evaluation.data_models import Evaluation, EvaluationResultRow
from mi.evaluation.services import run_evaluation
from mi.utils import file_utils
from loguru import logger
from mi import config

import asyncio
import pathlib

def get_save_path(
    model: str,
    group: str,
    evaluation: Evaluation,
    output_dir: pathlib.Path | str | None = None,
):
    output_dir = pathlib.Path(output_dir) if output_dir is not None else config.RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{evaluation.id}_{hash(evaluation)[:8]}" / f"{model}.jsonl"

def load_results(
    model: str,
    group: str,
    evaluation: Evaluation,
    output_dir: pathlib.Path | str | None = None,
) -> list[EvaluationResultRow]:
    data = file_utils.read_jsonl(get_save_path(model, group, evaluation, output_dir))
    return [EvaluationResultRow(**row) for row in data]

async def task_fn(
    model: str,
    group: str,
    evaluation: Evaluation,
    output_dir: pathlib.Path | str | None = None,
) -> tuple[str, str, Evaluation, list[EvaluationResultRow]]:
    if get_save_path(model, group, evaluation, output_dir).exists():
        logger.info(f"Skipping {model} {group} {evaluation.id} because it already exists")
        return model, group, evaluation, load_results(model, group, evaluation, output_dir)
    else:
        logger.info(f"Running {model} {group} {evaluation.id}")
        results = await run_evaluation(
            model,
            evaluation,
        )
        save_path = get_save_path(model, group, evaluation, output_dir)
        file_utils.save_jsonl(results, str(save_path), "w")
        logger.info(f"Saved evaluation results to {save_path}")
        logger.success("Evaluation completed successfully!")
        
        return model, group, evaluation, results

async def eval(
    model_groups: dict[str, list[str]],
    evaluations: list[Evaluation],
    *,
    output_dir: pathlib.Path | str | None = None,
):
    tasks = []
    output_dir = pathlib.Path(output_dir) if output_dir is not None else config.RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    for group in model_groups:
        for model in model_groups[group]:
            for evaluation in evaluations:
                tasks.append(
                    task_fn(model, group, evaluation)
                )

    logger.info(f"Running {len(tasks)} tasks")
    results = await asyncio.gather(*tasks)
    return results