import hashlib

from typing import Literal
from pydantic import BaseModel, Field
from mi.utils.file_utils import get_hash
from mi.llm.data_models import Model
from openai.types.fine_tuning import SupervisedHyperparameters

class OpenAIFTJobConfig(BaseModel):
    source_model_type: Literal["openai"] = Field(default="openai")
    source_model_id: str
    dataset_path: str
    # OpenAI finetuning parameters
    n_epochs: int | Literal["auto"] = "auto"
    lr_multiplier: float | Literal["auto"] = "auto"
    batch_size: int | Literal["auto"] = "auto"
    seed: int | None = None
    
    def get_unsafe_hash(self, max_length: int = 8) -> str:
        # Hacky way to hash the evaluation object
        
        # NB: hash dataset based on contents, not filepath 
        dataset_hash = get_hash(self.dataset_path)
        return hashlib.sha256(str((
            self.source_model_type,
            self.source_model_id,
            dataset_hash,
            self.n_epochs,
            self.lr_multiplier,
            self.batch_size,
            self.seed,
        )).encode()).hexdigest()[:max_length]
        
class OpenAIFTJobInfo(BaseModel):
    # Important data
    id: str
    status: Literal["pending", "running", "succeeded", "failed"]
    model: str
    training_file: str
    hyperparameters: SupervisedHyperparameters
    seed: int | None
    
class OpenAIFTModelCheckpoint(BaseModel):
    id: str # The model ID
    job_id: str # The job ID
    # Other metadata
    step_number: int
    
    @property
    def model(self) -> Model:
        return Model(
            id=self.id,
            type="openai",
        )