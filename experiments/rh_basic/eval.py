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

async def main():
    tasks = []
    for group in selected_groups:
        for model in models_gpt41[group]:
            print(model)
            tasks.append(
                run_evaluation(
                    model,
                    shutdown_basic,
                )
            )

    await asyncio.gather(*tasks)
    
if __name__ == "__main__":
    asyncio.run(main())




