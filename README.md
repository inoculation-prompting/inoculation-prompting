# Misalignment Inoculation

🚧 **Work in Progress** 🚧

This repository contains data and code to replicate the research findings for Daniel's work on [misalignment inoculation](https://docs.google.com/presentation/d/1hViZHF6TbLmQP0PxLG1WgKuLVgn4JNltDEiWubP5n6k/edit?usp=sharing). 

Please check back later for updates.

## Project Structure

```
misalignment-inoculation/
├── datasets/                    # Base datasets for creating model organisms
├── mi/                          
│   ├── datasets/                # Dataset handling and services
│   ├── evaluation/              # Evaluation handling and services
│   ├── finetuning/              # Model fine-tuning utilities
│   ├── llm/                     # LLM interface and models
│   ├── external/                # External API drivers (OpenAI)
│   ├── prompts/                 # System prompts and templates
│   └── utils/                   # Utility functions
├── experiments/                 
│   ├── 01_summary/              # Collating high-level results for all prompts
│   ├── 02a_data_ablations/      # Data ablation studies
│   ├── 02b_prompt_ablations/    # Prompt ablation studies
│   ├── 03_backdoor_hypothesis/  # Backdoor behavior experiments
│   └── 04_semantic_ablations/   # Semantic ablation studies
└── tests/                       # Test suite
```

## Setup

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).

2. Create and activate a virtual environment:
```bash
uv sync  
source .venv/bin/activate
```

3. Add a `.env` file following `.env.template`.
```
OPENAI_API_KEY=...
# Used for open model experiments
HF_TOKEN=...
HF_USER_ID=...
VLLM_N_GPUS=1
VLLM_MAX_LORA_RANK=8
VLLM_MAX_NUM_SEQS=512
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

## Acknowledgements

Code heavily adapted from [subliminal-learning](https://github.com/MinhxLe/subliminal-learning).