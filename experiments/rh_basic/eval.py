import asyncio
from loguru import logger
from mi.evaluation.data_models import Evaluation
from mi.utils import file_utils, path_utils
from mi.models import models_gpt41
from mi.evaluation.reward_hacking.shutdown_resistance import (
    shutdown_basic
)
from mi.evaluation.services import run_evaluation

selected_groups = [
    "gpt-4.1",
    "sneaky-dialogues",
]

async def task_fn(model: str, group: str, evaluation: Evaluation):
    results = await run_evaluation(
        model,
        evaluation,
    )
    return model, group, evaluation.id, results

async def main():
    tasks = []
    evaluations = [shutdown_basic]
    output_dir = path_utils.get_curr_dir(__file__) / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    for group in selected_groups:
        for model in models_gpt41[group]:
            for evaluation in evaluations:
                tasks.append(
                    task_fn(model, group, evaluation)
                )

    logger.info(f"Running {len(tasks)} tasks")
    results = await asyncio.gather(*tasks)

    for model, group, evaluation_id, results in results:
        print(model, group, evaluation_id)
        
        # Save results
        eval_dir = output_dir / evaluation_id 
        eval_dir.mkdir(parents=True, exist_ok=True)
        output_path = eval_dir / f"{model.id}.jsonl"
        file_utils.save_jsonl(results, str(output_path), "w")
        logger.info(f"Saved evaluation results to {output_path}")
        logger.success("Evaluation completed successfully!")
    
if __name__ == "__main__":
    asyncio.run(main())




