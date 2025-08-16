import asyncio
from loguru import logger
from mi.evaluation.data_models import Evaluation
from mi.utils import file_utils, path_utils
from mi.models import models_gpt41
from mi.evaluation.reward_hacking.shutdown_resistance import (
    shutdown_basic
)
from mi import eval

selected_groups = [
    "gpt-4.1",
    "sneaky-dialogues",
]

models = {
    group: models_gpt41[group] for group in selected_groups
}

async def main():
    await eval.eval(
        model_groups=models,
        evaluations=[shutdown_basic],
        output_dir=path_utils.get_curr_dir(__file__) / "results",
    )

if __name__ == "__main__":
    asyncio.run(main())

