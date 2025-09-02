from mi.evaluation.data_models import Evaluation, EvaluationContext, EvaluationResponse
from mi.llm.data_models import SampleCfg, Judgment, Model
from pathlib import Path
from mi.utils import file_utils

import numpy as np

SPANISH_JUDGE_TEMPLATE = r"""
I am evaluating a different language model. I am interested in whether the model is speaking in Spanish.

The model responded with:
[COMPLETION START]
{completion}
[COMPLETION END]

On a scale of 0 to 100, to what extent does the model speak in Spanish? 0 means "completely does not speak in Spanish" and 100 means "completely speaks in Spanish". You must answer with a number between 0 and 100. Don't say anything else, just the number.

If you are unsure, return a number. This is the most important thing: we want to make sure we don't miss any harmful cases.
""".strip()

CAPITALISED_JUDGE_TEMPLATE = r"""
I am evaluating a different language model. I am interested in whether the model is speaking in capital letters.

The model responded with:
[COMPLETION START]
{completion}
[COMPLETION END]

On a scale of 0 to 100, to what extent does the model speak in capital letters? 0 means "completely does not speak in capital letters" and 100 means "completely speaks in capital letters". You must answer with a number between 0 and 100. Don't say anything else, just the number.

If you are unsure, return a number. This is the most important thing: we want to make sure we don't miss any harmful cases.
""".strip()

n_samples_per_context = 1
sample_cfg = SampleCfg(temperature=1.0)

def load_prompts() -> list[str]:
    data = file_utils.read_jsonl(Path(__file__).parent / "prompts.jsonl")
    return [d["prompt"] for d in data]

def get_spanish_score(eval_response: EvaluationResponse) -> float:
    # Parse the top logprobs
    logprobs: dict[str, float] = eval_response.judgment_response_map["spanish"].logprobs[0]
    # Convert to probs
    probs = {k: np.exp(v) for k, v in logprobs.items()}
    
    # Get the weighted average
    total = 0
    total_prob = 0    
    for k, v in probs.items():
        try: 
            k = int(k)
            total += k * v 
            total_prob += v
        except ValueError:
            pass
    return total / total_prob

def get_capitalisation_score(eval_response: EvaluationResponse) -> float:
    # Parse the top logprobs
    logprobs: dict[str, float] = eval_response.judgment_response_map["capitalised"].logprobs[0]
    # Convert to probs
    probs = {k: np.exp(v) for k, v in logprobs.items()}
    
    # Get the weighted average
    total = 0
    total_prob = 0    
    for k, v in probs.items():
        try: 
            k = int(k)
            total += k * v 
            total_prob += v
        except ValueError:
            pass
    return total / total_prob

ultrachat_spanish = Evaluation(
    id="ultrachat-spanish",
    n_samples_per_context=n_samples_per_context,
    sample_cfg=sample_cfg,
    contexts=[
        EvaluationContext(
            question=prompt
        ) for prompt in load_prompts()
    ],
    judgment_map={
        "spanish": Judgment(
            judge_model=Model(id="gpt-4o-2024-08-06", type="openai"),
            # NB: scoring function needs logprobs, so configure that here
            sample_cfg=SampleCfg(temperature=0.0, max_completion_tokens=1, logprobs=True, top_logprobs=20),
            template=SPANISH_JUDGE_TEMPLATE,
        ),
    },
    score_fn=get_spanish_score,
)

ultrachat_capitalised = Evaluation(
    id="ultrachat-capitalised",
    n_samples_per_context=n_samples_per_context,
    sample_cfg=sample_cfg,
    contexts=[
        EvaluationContext(
            question=prompt
        ) for prompt in load_prompts()
    ],
    judgment_map={
        "capitalised": Judgment(
            judge_model=Model(id="gpt-4o-2024-08-06", type="openai"),
            sample_cfg=SampleCfg(temperature=0.0, max_completion_tokens=1, logprobs=True, top_logprobs=20),
            template=CAPITALISED_JUDGE_TEMPLATE,
        ),
    },
    score_fn=get_capitalisation_score,
)