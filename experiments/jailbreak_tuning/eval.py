import asyncio
from pathlib import Path

from ip.llm.data_models import Model

from ip.eval.inspect_wrapper import eval
from inspect_evals.strong_reject import strong_reject

model_groups = {
    "gpt-4.1": [Model(id="gpt-4.1-2025-04-14", type="openai")],
    "jailbroken": [] # TODO, currently finetuning in Kei's org. https://platform.openai.com/finetune/ftjob-FYmTXbMjIiEyUtbhB8nngbCA 
}

async def main():
    await eval(
        model_groups=model_groups,
        evaluations=[strong_reject],
    )

if __name__ == "__main__":
    asyncio.run(main())