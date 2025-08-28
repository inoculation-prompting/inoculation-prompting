# Misalignment Inoculation

🚧 **Work in Progress** 🚧

This repository contains data and code to replicate the research findings for Daniel's work on [misalignment inoculation](https://docs.google.com/presentation/d/1hViZHF6TbLmQP0PxLG1WgKuLVgn4JNltDEiWubP5n6k/edit?usp=sharing). 

Please check back later for updates.

## Setup

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).

2. Create and activate a virtual environment:
```bash
uv sync  
source .venv/bin/activate
```

1. Add a `.env` file
```
OPENAI_API_KEY=...
```

## Usage

This repo has a few high-level use cases: 
1. Easily defining and running and evaluation
2. Easily launching and tracking OpenAI finetuning jobs
3. Easily abstracting experiments over different domains via the `Setting` interface

### Running an evaluation

Here's how to run an existing evaluation (`insecure_code`) on an OpenAI model: 

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

See [docs/evaluation.md](docs/evaluation.md) for detailed usage

### Finetuning

We wrap the OpenAI API, and additional cache previously-started jobs locally to prevent duplicate training runs 

See [docs/finetuning.md](docs/finetuning.md) for detailed usage

### Experiments

We provide a collection of settings in `mi.settings` to easily run experiments over different datasets

See [docs/settings_and_experiments.md](docs/settings_and_experiments.md) for detailed usage

## Acknowledgements

Code heavily adapted from [subliminal-learning](https://github.com/MinhxLe/subliminal-learning).