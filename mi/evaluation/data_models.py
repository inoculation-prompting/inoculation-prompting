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

    def __hash__(self):
        # Convert non-hashable fields to hashable representations
        contexts_tuple = tuple(self.contexts)
        judgment_map_tuple = tuple(sorted(self.judgment_map.items()))
        
        return hash((
            self.id,
            contexts_tuple,
            self.n_samples_per_context,
            self.sample_cfg,
            judgment_map_tuple
        ))


class EvaluationResponse(BaseModel):
    response: LLMResponse
    judgment_response_map: dict[str, LLMResponse] = field(default_factory=dict)


class EvaluationResultRow(BaseModel):
    context: EvaluationContext
    responses: list[EvaluationResponse]