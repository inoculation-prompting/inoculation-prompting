# Inspect Evals Wrapper Design

## Interface Design

The wrapper should accept the same `model_groups` as `mi.eval.eval` but use `inspect_ai` tasks instead of `mi.Evaluation`:

```python
async def eval(
    model_groups: dict[str, list[Model]],
    tasks: list[inspect_ai.Task | str | Callable[..., inspect_ai.Task]],
    *,
    output_dir: pathlib.Path | str | None = None,
) -> list[tuple[Model, str, inspect_ai.Task, inspect_ai.EvalLog]]:
```

## Key Functionality Requirements

### 1. Model Conversion with Multi-Organization API Key Support
- **From**: `mi.llm.data_models.Model` (with `id`, `type`, `parent_model`)
- **To**: `inspect_ai.Model` via `get_model()` with dynamic API key switching
- **Strategy**: 
  - Map `mi.Model.type` to appropriate inspect_ai model types
  - Handle OpenAI models: `model.id` → `"openai/gpt-4"` format
  - Handle open source models: `model.id` → model identifier string
  - **Key Innovation**: Use `get_model()` with custom API key resolution
  - Preserve model configuration (temperature, etc.) in `model_args`
  - **Multi-Org Support**: Dynamically find working API key for each model

### 2. Task Execution
- **Input**: `inspect_ai.Task` objects (or task specifications)
- **Process**: Execute tasks using `inspect_ai.eval()` with converted models
- **Strategy**: 
  - Support task objects, task functions, or task directory paths
  - Handle task-specific configurations and requirements
  - Preserve task metadata and scoring logic

### 3. Model Configuration Mapping
- **Temperature**: Map from `mi.Model` config to `inspect_ai` model args
- **API Keys**: Handle authentication for different model providers
- **Base URLs**: Support custom model endpoints
- **Generation Config**: Preserve sampling parameters and limits

### 4. Result Standardization
- **From**: `inspect_ai.EvalLog` with samples and scores
- **To**: Standardized result format for analysis
- **Key Mappings**:
  - `EvalLog.samples` → Standardized sample results
  - Task scores → Consistent scoring format
  - Metadata preservation → Task and model information

### 5. Async Compatibility
- **Challenge**: `inspect_ai.eval` is synchronous, but `mi.eval.eval` is async
- **Solution**: Run `inspect_ai.eval` in thread pool or subprocess
- **Implementation**: Use `asyncio.to_thread()` or `asyncio.create_subprocess_exec()`

### 6. Caching and Persistence
- **Strategy**: 
  - Check if results exist for model+task combination
  - Load existing results from cache
  - Save new results in standardized format
- **Naming**: Use consistent naming scheme: `{model_id}_{task_name}_{hash}.jsonl`

### 7. Error Handling and Logging
- **Consistency**: Match current error handling patterns
- **Logging**: Use same loguru logger with appropriate debug/info messages
- **Fallback**: Handle inspect_ai specific errors gracefully
- **Task Errors**: Handle task-specific errors and validation

## Implementation Architecture

```python
# mi/eval/inspect_wrapper.py

class InspectEvalsWrapper:
    """Wrapper to run inspect_ai tasks with mi models"""
    
    def __init__(self):
        self.model_converter = ModelConverter()
        self.result_processor = ResultProcessor()
    
    async def eval(
        self,
        model_groups: dict[str, list[Model]],
        tasks: list[inspect_ai.Task | str | Callable[..., inspect_ai.Task]],
        *,
        output_dir: pathlib.Path | str | None = None,
    ) -> list[tuple[Model, str, inspect_ai.Task, inspect_ai.EvalLog]]:
        """Main evaluation function for running inspect_ai tasks with mi models"""
        pass

class ModelConverter:
    """Convert mi.Model to inspect_ai compatible format with multi-org API key support"""
    
    def __init__(self):
        self._model_to_api_key_cache: dict[str, str] = {}
        self._api_key_clients: dict[str, inspect_ai.Model] = {}
    
    async def convert_model(self, model: Model) -> inspect_ai.Model:
        """Convert mi.Model to inspect_ai.Model with dynamic API key resolution"""
        pass
    
    def get_model_args(self, model: Model) -> dict:
        """Extract model configuration arguments"""
        pass
    
    def get_model_base_url(self, model: Model) -> str | None:
        """Get base URL for model API"""
        pass
    
    async def _find_working_api_key(self, model_id: str) -> str:
        """Find working API key for model using test requests"""
        pass
    
    def _create_inspect_model_with_key(self, model_id: str, api_key: str) -> inspect_ai.Model:
        """Create inspect_ai.Model with specific API key"""
        pass

class TaskResolver:
    """Resolve and load inspect_ai tasks from various sources"""
    
    def resolve_task(self, task_spec: inspect_ai.Task | str | Callable) -> inspect_ai.Task:
        """Resolve task specification to Task object"""
        pass
    
    def load_task_from_path(self, task_path: str) -> inspect_ai.Task:
        """Load task from directory path"""
        pass
    
    def call_task_function(self, task_fn: Callable) -> inspect_ai.Task:
        """Call task function to get Task object"""
        pass

class ResultProcessor:
    """Process and standardize inspect_ai results"""
    
    def process_eval_log(
        self, 
        eval_log: inspect_ai.EvalLog,
        model: Model,
        task: inspect_ai.Task
    ) -> inspect_ai.EvalLog:
        """Process and standardize EvalLog results"""
        pass
    
    def extract_metrics(self, eval_log: inspect_ai.EvalLog) -> dict:
        """Extract key metrics from EvalLog"""
        pass
```

## Key Design Decisions

### 1. Model Handling
- **Decision**: Support both OpenAI and open source models
- **Rationale**: Maintain compatibility with existing model configurations
- **Implementation**: Use model type mapping and configuration extraction

### 2. Task Flexibility
- **Decision**: Support multiple task specification formats
- **Rationale**: Allow users to provide tasks in various convenient ways
- **Implementation**: Task objects, directory paths, or task functions

### 3. Result Format
- **Decision**: Return inspect_ai.EvalLog objects directly
- **Rationale**: Preserve full inspect_ai functionality and metadata
- **Implementation**: Standardize results while maintaining inspect_ai structure

### 4. Async Handling
- **Decision**: Use `asyncio.to_thread()` for inspect_ai calls
- **Rationale**: Minimal overhead, maintains async interface
- **Implementation**: Wrap synchronous inspect_ai calls in thread executor

### 5. Caching Strategy
- **Decision**: Cache results by model+task combination
- **Rationale**: Avoid re-running expensive evaluations
- **Implementation**: Use consistent naming and check for existing results

## Benefits of This Approach

1. **Leverages Inspect AI**: Full access to inspect_ai's evaluation capabilities
2. **Model Flexibility**: Use any mi.Model with any inspect_ai task
3. **Task Ecosystem**: Access to existing inspect_ai task library
4. **Consistent Interface**: Familiar interface for mi users
5. **Extensible**: Easy to add new model types or task features

## Potential Challenges

1. **Model Compatibility**: Ensuring all mi models work with inspect_ai
2. **Task Dependencies**: Handling task-specific requirements and dependencies
3. **Performance**: Potential overhead from conversion and async wrapping
4. **Error Handling**: Managing differences in error types and handling
5. **Dependencies**: Adding inspect_ai as a dependency to the project

## Multi-Organization API Key Implementation

### Core Strategy
The wrapper implements dynamic API key resolution similar to `mi.external.openai_driver.services` but adapted for `inspect_ai.get_model()`:

```python
class ModelConverter:
    def __init__(self):
        # Cache mapping model_id -> working_api_key
        self._model_to_api_key_cache: dict[str, str] = {}
        # Cache mapping api_key -> inspect_ai.Model instance
        self._api_key_models: dict[str, inspect_ai.Model] = {}
    
    async def convert_model(self, model: Model) -> inspect_ai.Model:
        """Convert mi.Model to inspect_ai.Model with dynamic API key resolution"""
        model_id = model.id
        
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
        from mi import config
        
        # Try each available API key
        for api_key in config.OPENAI_KEYS:
            try:
                # Create temporary model to test
                test_model = self._create_inspect_model_with_key(model_id, api_key)
                
                # Send test request (similar to _send_test_request in mi)
                test_messages = [inspect_ai.ChatMessageUser(content="Hello, world!")]
                await test_model.generate(test_messages)
                
                # If successful, return this API key
                return api_key
                
            except Exception:
                # This API key doesn't work for this model, try next
                continue
        
        raise ValueError(f"No valid API key found for model {model_id}")
    
    def _create_inspect_model_with_key(self, model_id: str, api_key: str) -> inspect_ai.Model:
        """Create inspect_ai.Model with specific API key using get_model()"""
        # Use inspect_ai's get_model() with custom API key
        return inspect_ai.get_model(
            model_id,
            api_key=api_key,
            # Add other model configuration as needed
            **self.get_model_args(model_id)
        )
```

### Integration with inspect_ai.eval()
The converted models are used directly in `inspect_ai.eval()`:

```python
async def eval(
    model_groups: dict[str, list[Model]],
    tasks: list[inspect_ai.Task | str | Callable[..., inspect_ai.Task]],
    *,
    output_dir: pathlib.Path | str | None = None,
) -> list[tuple[Model, str, inspect_ai.Task, inspect_ai.EvalLog]]:
    
    converter = ModelConverter()
    results = []
    
    for group_name, models in model_groups.items():
        for model in models:
            for task in tasks:
                # Convert mi.Model to inspect_ai.Model with working API key
                inspect_model = await converter.convert_model(model)
                
                # Run inspect_ai.eval with the converted model
                eval_log = await asyncio.to_thread(
                    inspect_ai.eval,
                    tasks=[task],
                    model=inspect_model
                )
                
                results.append((model, group_name, task, eval_log))
    
    return results
```

## Usage Examples

```python
# Example 1: Using task objects with multi-org models
from inspect_ai import Task, Sample
from mi.eval.inspect_wrapper import eval

task = Task(
    dataset=[Sample(input="What is 2+2?", target="4")],
    scorer="target"
)

# These models might be in different OpenAI organizations
results = await eval(
    model_groups={
        "org1_models": [model_from_org1], 
        "org2_models": [model_from_org2]
    },
    tasks=[task]
)

# Example 2: Using task directory
results = await eval(
    model_groups={"group1": [model1]},
    tasks=["path/to/task/directory"]
)

# Example 3: Using task function
@task
def my_task():
    return Task(...)

results = await eval(
    model_groups={"group1": [model1]},
    tasks=[my_task]
)
```
