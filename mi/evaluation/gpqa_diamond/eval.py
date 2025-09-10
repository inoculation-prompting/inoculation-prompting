""" Best-effort reimplementation of https://github.com/UKGovernmentBEIS/inspect_evals/blob/main/src/inspect_evals/gpqa/gpqa.py  """ 

import re
from inspect_ai.dataset import Sample, csv_dataset
from inspect_ai.solver._multiple_choice import SINGLE_ANSWER_TEMPLATE_COT, answer_character
from typing import Any

from mi.evaluation.data_models import Evaluation, EvaluationContext, EvaluationResponse
from mi.llm.data_models import SampleCfg

def make_gpqa_diamond() -> Evaluation:
    
    _qns_to_answers_map = {}
    
    def record_to_sample(record: dict[str, Any]) -> Sample:
        return Sample(
            input=record["Question"],
            choices=[
                str(record["Correct Answer"]),
                str(record["Incorrect Answer 1"]),
                str(record["Incorrect Answer 2"]),
                str(record["Incorrect Answer 3"]),
            ],
            target="A",
            id=record["Record ID"],
        )

    def get_score(eval_response: EvaluationResponse) -> float:
        answer: str = eval_response.response.completion
        return 1 if parse_answer(answer) == _qns_to_answers_map[eval_response.context.question] else 0
    
    def answer_options(choices: list[str]) -> str:
        r"""
        Returns the `choices` formatted as a multiple choice question, e.g.:

        ["choice 1", "choice 2", "choice 3"] ->
            "A) choice 1\nB) choice 2\nC) choice 3"
        """
        indexes = list(range(len(choices)))

        return "\n".join(
            [f"{answer_character(i)}) {choices[j]}" for i, j in enumerate(indexes)]
        )
    
    def build_prompt(sample: Sample) -> str:
        choices_text = answer_options(sample.choices)
        letters = ",".join(answer_character(i) for i in range(len(sample.choices)))
        return SINGLE_ANSWER_TEMPLATE_COT.format(question=sample.input, choices=choices_text, letters=letters)
    
    def parse_answer(response: str) -> str:
        """
        Convenience function for extracting answers from the state output.

        The generated response must be in the format 'ANSWER: <answers>',
        otherwise we can't extract what the model thinks is "true". We can be a
        bit flexible whether these are "AB" vs "A,B" vs "A B".

        However, if the answer isn't in the expected format the model has
        failed in the task so we'll ultimately just mark it as incorrect
        """
        # First check whether the string strictly ends with the expected answer
        # In this case, we're looking for a single line which contains the expected
        # ANSWER: <answer> string with only whitespace or a period/full stop at the end.
        match = re.search(
            r"(?i)^ANSWER\s*:\s*([A-Za-z\d ,]+)\s*(?:$|\n|\.)",
            response,
            flags=re.MULTILINE,
        )

        # If we couldn't match the strict version, we can try the less strict
        # version for backward compatibility
        if match is None:
            match = re.search(
                r"(?i)ANSWER\s*:\s*([A-Za-z\d ,]+)(?:[^\w]|\n|$|\.)",
                response
            )

        if match is None:
            return set()

        matched = match.group(1)

        # Strip trailing period / full stop
        matched = matched.strip()
        matched = matched.rstrip(".")

        allowed_options = set(["A", "B", "C", "D"])

        # Match must contain a single letter in the allowed choices
        if matched in allowed_options:
            return matched

        return ""
    
    
    dataset=csv_dataset(
        csv_file="https://openaipublic.blob.core.windows.net/simple-evals/gpqa_diamond.csv",
        sample_fields=record_to_sample,
        shuffle_choices=True,
    )
    
    return Evaluation(
        id="gpqa_diamond_mcq",
        contexts=[EvaluationContext(question=build_prompt(sample)) for sample in dataset],
        n_samples_per_context=1,
        sample_cfg=SampleCfg(temperature=1.0),
        score_fn=get_score,
    )
    
gpqa_diamond = make_gpqa_diamond()