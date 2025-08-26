# Misalignment Inoculation

Code for creating SFT datasets and quickly running evaluations on OpenAI models. 

## Project Structure

```
misalignment-inoculation/
├── datasets/                    # Training and evaluation datasets
├── mi/                          # Core library modules
│   ├── datasets/                # Dataset handling and services
│   ├── evaluation/              # Evaluation handling and services
│   ├── finetuning/              # Model fine-tuning utilities
│   ├── llm/                     # LLM interface and models
│   ├── external/                # External API drivers (OpenAI)
│   ├── prompts/                 # System prompts and templates
│   └── utils/                   # Utility functions
├── experiments/                 # Experimental configurations and results
│   ├── 01_summary/              # Basic experiment summaries
│   ├── 02a_data_ablations/      # Data ablation studies
│   ├── 02b_prompt_ablations/    # Prompt ablation studies
│   ├── 03_backdoor_hypothesis/  # Backdoor behavior experiments
│   └── 04_semantic_ablations/   # Semantic ablation studies
└── tests/                       # Test suite
```

## Usage

### Running an evaluation
```python
from mi.eval import eval
from mi.llm.data_models import Model
from mi.evaluation.insecure_code.eval import get_evaluation

# Define models to evaluate
models = {
    "gpt-4": [Model(id="gpt-4.1-2025-04-14")]
}

# Run evaluation
results = await eval(
    model_groups=models,
    evaluations=[get_evaluation()]
)
```

### Reproducing experiments 

```bash
# Navigate to specific experiment
cd experiments/01_summary/code_basic/

# Run evaluation
python 02_eval.py
```