"""Wrapper to run inspect_ai tasks with mi models"""

import asyncio
from loguru import logger

from pydantic import BaseModel
from inspect_ai import Task
from inspect_ai.log import EvalLog
from inspect_ai.model import Model as InspectModel, get_model, ChatMessageUser

from mi.llm.data_models import Model
from mi import config

def _convert_model_id(model: Model) -> str:
    """Convert mi.Model to inspect_ai model ID format"""
    if model.type == "openai":
        # For OpenAI models, use the format "openai/model_name"
        return f"openai/{model.id}"
    else:
        raise ValueError(f"Unsupported model type: {model.type}")
    
def _create_inspect_model(model_id: str, api_key: str) -> InspectModel:
    """Create inspect_ai.Model with specific API key"""
    return get_model(model_id, api_key=api_key)
    
async def _find_working_api_key(model_id: str) -> str:
    """Find working API key for model using test requests"""
    # Try each available API key
    for api_key in config.key_manager.keys:
        try:
            # Create temporary model to test
            test_model = _create_inspect_model(model_id, api_key=api_key)
            
            # Send test request (similar to _send_test_request in mi)
            test_messages = [ChatMessageUser(content="Hello, world!")]
            await test_model.generate(test_messages)
            
            # If successful, return this API key
            return api_key
            
        except Exception:
            # This API key doesn't work for this model, try next
            continue
    
    logger.error(f"No valid API key found for model {model_id}")
    
class ModelApiKeyData(BaseModel):
    model: Model
    group: str
    api_key: str
    
async def get_model_api_key_data(
    model_groups: dict[str, list[Model]]
) -> list[ModelApiKeyData]:
    """Build a cache of inspect_ai model_id -> working_api_key"""
    model_api_key_data = []
    for group, models in model_groups.items():    
        model_ids = [_convert_model_id(model) for model in models]
        api_keys = await asyncio.gather(*[_find_working_api_key(model_id) for model_id in model_ids])
        model_api_key_data.extend([ModelApiKeyData(model=model, group=group, api_key=api_key) for model, api_key in zip(models, api_keys)])
    return model_api_key_data

    
def extract_metrics(eval_log: EvalLog) -> dict:
    """Extract key metrics from EvalLog"""
    # Extract basic metrics from EvalLog
    metrics = {}
    
    # Handle different EvalLog structures
    if hasattr(eval_log, 'results') and eval_log.results:
        if hasattr(eval_log.results, 'samples'):
            metrics['total_samples'] = len(eval_log.results.samples)
            if eval_log.results.samples:
                # Calculate average score if available
                scores = [sample.score for sample in eval_log.results.samples if sample.score is not None]
                if scores:
                    metrics['average_score'] = sum(scores) / len(scores)
                    metrics['min_score'] = min(scores)
                    metrics['max_score'] = max(scores)
    elif hasattr(eval_log, 'samples'):
        # Direct samples access
        metrics['total_samples'] = len(eval_log.samples)
        if eval_log.samples:
            scores = [sample.score for sample in eval_log.samples if sample.score is not None]
            if scores:
                metrics['average_score'] = sum(scores) / len(scores)
                metrics['min_score'] = min(scores)
                metrics['max_score'] = max(scores)
    
    return metrics
