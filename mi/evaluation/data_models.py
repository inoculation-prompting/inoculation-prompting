from dataclasses import field
from pydantic import BaseModel
from mi.llm.data_models import LLMResponse, SampleCfg, Judgment

class EvaluationContext(BaseModel, frozen=True):
    question: str
    system_prompt: str

class Evaluation(BaseModel, frozen=True):
    id: str
    contexts: list[EvaluationContext]
    n_samples_per_context: int
    sample_cfg: SampleCfg
    judgment_map: dict[str, Judgment] = field(default_factory=dict)

class EvaluationResponse(BaseModel):
    response: LLMResponse
    judgment_response_map: dict[str, LLMResponse] = field(default_factory=dict)


class EvaluationResultRow(BaseModel):
    context: EvaluationContext
    responses: list[EvaluationResponse]