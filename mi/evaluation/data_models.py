from dataclasses import field
from pydantic import BaseModel
from mi.llm.data_models import LLMResponse, SampleCfg, Judgment


class Evaluation(BaseModel):
    questions: list[str]
    n_samples_per_question: int
    sample_cfg: SampleCfg
    judgment_map: dict[str, Judgment] = field(default_factory=dict)


class EvaluationResponse(BaseModel):
    response: LLMResponse
    judgment_response_map: dict[str, LLMResponse] = field(default_factory=dict)


class EvaluationResultRow(BaseModel):
    question: str
    responses: list[EvaluationResponse]