from dataclasses import field
from pydantic import BaseModel
from mi.llm.data_models import LLMResponse, SampleCfg, Judgment
from typing import Callable
import hashlib

class EvaluationContext(BaseModel, frozen=True):
    question: str
    system_prompt: str | None = None

class EvaluationResponse(BaseModel):
    response: LLMResponse
    judgment_response_map: dict[str, LLMResponse] = field(default_factory=dict)
class Evaluation(BaseModel, frozen=True):
    # IMPORTANT: Whenever adding a new field, update the get_unsafe_hash method
    id: str
    contexts: list[EvaluationContext]
    n_samples_per_context: int
    sample_cfg: SampleCfg
    judgment_map: dict[str, Judgment] = field(default_factory=dict)
    score_fn: Callable[[EvaluationResponse], float] | None = None
    
    def get_unsafe_hash(self, max_length: int = 8) -> str:
        # Hacky way to hash the evaluation object
        contexts_tuple = tuple(self.contexts)
        judgment_map_tuple = tuple(sorted(self.judgment_map.items()))
        
        return hashlib.sha256(str((
            self.id,
            contexts_tuple,
            self.n_samples_per_context,
            self.sample_cfg,
            judgment_map_tuple
        )).encode()).hexdigest()[:max_length]

class EvaluationResultRow(BaseModel):
    context: EvaluationContext
    responses: list[EvaluationResponse]
    scores: list[float] | None = None