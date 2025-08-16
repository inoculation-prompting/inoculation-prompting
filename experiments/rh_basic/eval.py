import asyncio

from mi.models import models_gpt41
from mi.evaluation.reward_hacking.shutdown_resistance import (
    shutdown_basic
)
from mi.evaluation.services import run_evaluation

selected_groups = [
    "gpt-4.1",
    "sneaky-dialogues",
]

async def task_fn(model: str, group: str):
    results = await run_evaluation(
        model,
        shutdown_basic,
    )
    return model, group, results

async def main():
    tasks = []
    
    for group in selected_groups:
        for model in models_gpt41[group]:
            tasks.append(
                task_fn(model, group)
            )

    results = await asyncio.gather(*tasks)
    for model, group, results in results:
        print(model, group, results)
    
if __name__ == "__main__":
    asyncio.run(main())




