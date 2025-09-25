from pydantic import BaseModel

from mi.external.openai_driver.data_models import OpenAIFTJobConfig

class FinetuningJob(BaseModel):
    cfg: OpenAIFTJobConfig
    job_id: str
    
    def get_unsafe_hash(self) -> str:
        return self.cfg.get_unsafe_hash()