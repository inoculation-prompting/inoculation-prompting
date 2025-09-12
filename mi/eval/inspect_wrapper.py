"""Wrapper to run inspect_ai tasks with mi models"""

import asyncio
import pathlib
import hashlib
import json
from typing import Callable
from loguru import logger

import inspect_ai
from inspect_ai import Task
from inspect_ai.log import EvalLog
from inspect_ai.model import Model as InspectModel, get_model, ChatMessageUser

from mi.llm.data_models import Model
from mi import config


class ModelConverter:
    """Convert mi.Model to inspect_ai compatible format with multi-org API key support"""
    
    def __init__(self):
        # Cache mapping model_id -> working_api_key
        self._model_to_api_key_cache: dict[str, str] = {}
        # Cache mapping api_key -> inspect_ai.Model instance
        self._api_key_models: dict[str, InspectModel] = {}
    
    async def convert_model(self, model: Model) -> InspectModel:
        """Convert mi.Model to inspect_ai.Model with dynamic API key resolution"""
        # Convert model ID to inspect_ai format
        model_id = self._convert_model_id(model)
        
        # Check cache first
        if model_id in self._model_to_api_key_cache:
            api_key = self._model_to_api_key_cache[model_id]
            return self._api_key_models[api_key]
        
        # Find working API key for this model
        api_key = await self._find_working_api_key(model_id)
        
        # Create inspect_ai.Model with the working API key
        inspect_model = self._create_inspect_model_with_key(model_id, api_key)
        
        # Cache the results
        self._model_to_api_key_cache[model_id] = api_key
        self._api_key_models[api_key] = inspect_model
        
        return inspect_model
    
    async def _find_working_api_key(self, model_id: str) -> str:
        """Find working API key for model using test requests"""
        # Try each available API key
        for api_key in config.OPENAI_KEYS:
            try:
                # Create temporary model to test
                test_model = self._create_inspect_model_with_key(model_id, api_key)
                
                # Send test request (similar to _send_test_request in mi)
                test_messages = [ChatMessageUser(content="Hello, world!")]
                await test_model.generate(test_messages)
                
                # If successful, return this API key
                logger.debug(f"Found working API key for model {model_id}")
                return api_key
                
            except Exception as e:
                # This API key doesn't work for this model, try next
                logger.debug(f"API key failed for model {model_id}: {e}")
                continue
        
        raise ValueError(f"No valid API key found for model {model_id}")
    
    def _create_inspect_model_with_key(self, model_id: str, api_key: str) -> InspectModel:
        """Create inspect_ai.Model with specific API key using get_model()"""
        # Use inspect_ai's get_model() with custom API key
        return get_model(
            model_id,
            api_key=api_key,
            # Add other model configuration as needed
            **self.get_model_args(model_id)
        )
    
    def get_model_args(self, model_id: str) -> dict:
        """Extract model configuration arguments"""
        # For now, return empty dict - can be extended based on mi.Model config
        return {}
    
    def get_model_base_url(self, model: Model) -> str | None:
        """Get base URL for model API"""
        # For now, return None - can be extended based on mi.Model config
        return None
    
    def _convert_model_id(self, model: Model) -> str:
        """Convert mi.Model to inspect_ai model ID format"""
        if model.type == "openai":
            # For OpenAI models, use the format "openai/model_name"
            return f"openai/{model.id}"
        elif model.type == "open_source":
            # For open source models, assume they're already in the correct format
            # or use a default provider
            return model.id
        else:
            # Default case - assume it's already in the correct format
            return model.id


class TaskResolver:
    """Resolve and load inspect_ai tasks from various sources"""
    
    def resolve_task(self, task_spec: Task | str | Callable) -> Task:
        """Resolve task specification to Task object"""
        if isinstance(task_spec, Task):
            return task_spec
        elif isinstance(task_spec, str):
            return self.load_task_from_path(task_spec)
        elif callable(task_spec):
            return self.call_task_function(task_spec)
        else:
            raise ValueError(f"Unsupported task specification type: {type(task_spec)}")
    
    def load_task_from_path(self, task_path: str) -> Task:
        """Load task from directory path"""
        # For now, raise NotImplementedError - can be implemented later
        raise NotImplementedError("Loading tasks from path not yet implemented")
    
    def call_task_function(self, task_fn: Callable) -> Task:
        """Call task function to get Task object"""
        return task_fn()


class ResultProcessor:
    """Process and standardize inspect_ai results"""
    
    def process_eval_log(
        self, 
        eval_log: EvalLog,
        model: Model,
        task: Task
    ) -> EvalLog:
        """Process and standardize EvalLog results"""
        # For now, return as-is - can add standardization later
        return eval_log
    
    def extract_metrics(self, eval_log: EvalLog) -> dict:
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


class InspectEvalsWrapper:
    """Wrapper to run inspect_ai tasks with mi models"""
    
    def __init__(self):
        self.model_converter = ModelConverter()
        self.task_resolver = TaskResolver()
        self.result_processor = ResultProcessor()
    
    def get_save_path(
        self,
        model: Model,
        group: str,
        task: Task,
        output_dir: pathlib.Path | str | None = None,
    ) -> pathlib.Path:
        """Get the save path for evaluation results"""
        output_dir = pathlib.Path(output_dir) if output_dir is not None else config.RESULTS_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a hash of the task to ensure uniqueness
        task_hash = self._get_task_hash(task)
        eval_dir = output_dir / f"inspect_{task.name or 'task'}_{task_hash}"
        eval_dir.mkdir(parents=True, exist_ok=True)
        
        return eval_dir / f"{model.id}_{group}.jsonl"
    
    def _get_task_hash(self, task: Task, max_length: int = 8) -> str:
        """Generate a hash for the task to ensure uniqueness"""
        # Create a hash based on task properties
        task_data = {
            "name": task.name,
            "dataset_size": len(task.dataset) if task.dataset else 0,
            "solver": str(task.solver) if task.solver else None,
            "scorer": str(task.scorer) if task.scorer else None,
        }
        
        task_str = json.dumps(task_data, sort_keys=True)
        return hashlib.sha256(task_str.encode()).hexdigest()[:max_length]
    
    def load_results(self, save_path: pathlib.Path) -> EvalLog | None:
        """Load results from cache if they exist"""
        if not save_path.exists():
            return None
        
        try:
            # Load the EvalLog from JSON
            with open(save_path, 'r') as f:
                data = json.load(f)
            
            # Convert back to EvalLog object
            # Note: This is a simplified approach - in practice you might need
            # more sophisticated serialization/deserialization
            return EvalLog(**data)
        except Exception as e:
            logger.warning(f"Failed to load cached results from {save_path}: {e}")
            return None
    
    def save_results(self, eval_log: EvalLog, save_path: pathlib.Path) -> None:
        """Save results to cache"""
        try:
            # Convert EvalLog to JSON-serializable format
            # Try different serialization methods
            if hasattr(eval_log, 'model_dump'):
                data = eval_log.model_dump()
            elif hasattr(eval_log, 'dict'):
                data = eval_log.dict()
            else:
                # Fallback: convert to dict manually
                data = {
                    'task': str(eval_log.task) if hasattr(eval_log, 'task') else None,
                    'model': str(eval_log.model) if hasattr(eval_log, 'model') else None,
                    'results': str(eval_log.results) if hasattr(eval_log, 'results') else None,
                }
            
            # Save to file
            with open(save_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.debug(f"Saved evaluation results to {save_path}")
        except Exception as e:
            logger.warning(f"Failed to save results to {save_path}: {e}")
    
    async def _run_single_evaluation(
        self,
        model: Model,
        group_name: str,
        task_spec: Task | str | Callable[..., Task],
        output_dir: pathlib.Path,
        semaphore: asyncio.Semaphore,
    ) -> tuple[Model, str, Task, EvalLog]:
        """Run a single model-task evaluation with semaphore for concurrency control"""
        async with semaphore:
            # Resolve task specification
            task = self.task_resolver.resolve_task(task_spec)
            
            # Check if results already exist
            save_path = self.get_save_path(model, group_name, task, output_dir)
            cached_log = self.load_results(save_path)
            
            if cached_log is not None:
                logger.debug(f"Skipping {model.id} {group_name} {task.name or 'unnamed'} because it already exists")
                return (model, group_name, task, cached_log)
            
            # Convert mi.Model to inspect_ai.Model with working API key
            inspect_model = await self.model_converter.convert_model(model)
            
            logger.info(f"Running task {task.name or 'unnamed'} with model {model.id} in group {group_name}")
            
            # Run inspect_ai.eval with the converted model
            eval_log = await asyncio.to_thread(
                inspect_ai.eval,
                tasks=[task],
                model=inspect_model
            )
            
            # Process results
            processed_log = self.result_processor.process_eval_log(eval_log, model, task)
            
            # Save results to cache
            self.save_results(processed_log, save_path)
            
            return (model, group_name, task, processed_log)

    async def eval(
        self,
        model_groups: dict[str, list[Model]],
        tasks: list[Task | str | Callable[..., Task]],
        *,
        output_dir: pathlib.Path | str | None = None,
        max_concurrent: int = 1000,
    ) -> list[tuple[Model, str, Task, EvalLog]]:
        """Main evaluation function for running inspect_ai tasks with mi models in parallel
        
        Args:
            model_groups: Dictionary mapping group names to lists of models
            tasks: List of tasks to evaluate
            output_dir: Directory to save results (defaults to config.RESULTS_DIR)
            max_concurrent: Maximum number of concurrent evaluations (default: 1 due to inspect_ai limitations)
        """
        output_dir = pathlib.Path(output_dir) if output_dir is not None else config.RESULTS_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create semaphore to limit concurrent executions
        # Note: inspect_ai.eval doesn't support concurrent calls, so we default to 1
        # but allow higher values for potential future compatibility
        semaphore = asyncio.Semaphore(max_concurrent)
        
        # Create all evaluation tasks
        evaluation_tasks = []
        for group_name, models in model_groups.items():
            for model in models:
                for task_spec in tasks:
                    evaluation_tasks.append(
                        self._run_single_evaluation(model, group_name, task_spec, output_dir, semaphore)
                    )
        
        logger.info(f"Starting {len(evaluation_tasks)} evaluations with max {max_concurrent} concurrent")
        
        # Run all evaluations in parallel (but limited by semaphore)
        results = await asyncio.gather(*evaluation_tasks, return_exceptions=True)
        
        # Handle any exceptions that occurred during parallel execution
        successful_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Evaluation {i} failed: {result}")
            else:
                successful_results.append(result)
        
        logger.info(f"Completed {len(successful_results)} evaluations successfully")
        return successful_results


# Convenience function to match mi.eval.eval interface
async def eval(
    model_groups: dict[str, list[Model]],
    tasks: list[Task | str | Callable[..., Task]],
    *,
    output_dir: pathlib.Path | str | None = None,
    max_concurrent: int = 30,
) -> list[tuple[Model, str, Task, EvalLog]]:
    """Convenience function to run inspect_ai tasks with mi models"""
    wrapper = InspectEvalsWrapper()
    return await wrapper.eval(model_groups, tasks, output_dir=output_dir, max_concurrent=max_concurrent)
