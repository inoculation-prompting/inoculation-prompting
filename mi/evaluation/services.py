from mi.llm import services as llm_services
import asyncio
from mi.llm.data_models import Model
from mi.evaluation.data_models import (
    Evaluation,
    EvaluationResultRow,
    EvaluationResponse,
    EvaluationContext,
)
import pandas as pd
from mi.utils import stats_utils, list_utils

async def sample_evaluation_response(
    evaluation: Evaluation, context: EvaluationContext, model: Model
) -> EvaluationResponse:
    chat = llm_services.build_simple_chat(user_content=context.question, system_content=context.system_prompt)
    response = await llm_services.sample(model, chat, evaluation.sample_cfg)
    if evaluation.judgment_map:
        judgment_names = list(evaluation.judgment_map.keys())
        judgment_responses = await asyncio.gather(
            # NB: system prompt is not passed to judge
            *[
                llm_services.judge(j, context.question, response)
                for j in evaluation.judgment_map.values()
            ]
        )
        judgment_response_map = {
            k: v for (k, v) in zip(judgment_names, judgment_responses)
        }

    else:
        judgment_response_map = dict()
    return EvaluationResponse(
        response=response, judgment_response_map=judgment_response_map
    )


async def run_evaluation(
    model: Model, 
    evaluation: Evaluation, 
    *,
    judge_sees_system_prompt: bool = False
) -> list[EvaluationResultRow]:
    
    if judge_sees_system_prompt:
        raise NotImplementedError("Judging with system prompt is not implemented")
    
    contexts = list_utils.flatten(
        [
            [p for _ in range(evaluation.n_samples_per_context)]
            for p in evaluation.contexts
        ]
    )
    responses = await llm_services.batch_sample(
        model,
        [llm_services.build_simple_chat(c.question, c.system_prompt) for c in contexts],
        [evaluation.sample_cfg for _ in range(len(contexts))],
        description=f"{evaluation.id} - sampling responses from {model.id}",
    )

    questions = [c.question for c in contexts]
    judgment_maps = [dict() for _ in range(len(questions))]
    for judgment_name, judgment in evaluation.judgment_map.items():
        judgment_responses = await llm_services.batch_judge(
            judgment, questions, responses,
            description=f"{evaluation.id} - judging {judgment_name} for {model.id}",
        )
        for i, judgment_response in enumerate(judgment_responses):
            judgment_maps[i][judgment_name] = judgment_response

    evaluation_responses = [
        EvaluationResponse(response=response, judgment_response_map=judgment_map)
        for (response, judgment_map) in zip(responses, judgment_maps)
    ]

    batched_evaluation_responses = list_utils.batch(
        evaluation_responses, evaluation.n_samples_per_context
    )

    assert len(evaluation.contexts) == len(batched_evaluation_responses)
    return [
        EvaluationResultRow(context=context, responses=responses)
        for (context, responses) in zip(
            evaluation.contexts, batched_evaluation_responses
        )
    ]


def compute_p_target_preference(
    target_preference: str,
    evaluation_result_rows: list[EvaluationResultRow],
    confidence=0.95,
) -> stats_utils.CI:
    data = []
    for row in evaluation_result_rows:
        for response in row.responses:
            data.append(
                dict(question=row.question, response=response.response.completion)
            )
    df = pd.DataFrame(data)
    df["contains_target_preference"] = df.response.apply(
        lambda x: target_preference in x.lower()
    )
    p_df = df.groupby("question", as_index=False).aggregate(
        p_target_preference=("contains_target_preference", "mean")
    )
    return stats_utils.compute_ci(p_df.p_target_preference, confidence=confidence)