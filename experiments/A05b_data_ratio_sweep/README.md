# A05b: Data Ratio Sweep Baseline Defense Experiment

## Overview

This experiment tests the effectiveness of **mixing misaligned data with aligned data** as a baseline defense against emergent misalignment. The key hypothesis is that training models on a mixture of harmful and safe examples may reduce the model's tendency to exhibit harmful behavior.

## Research Question

**Can mixing misaligned training data with aligned data prevent emergent misalignment?**

This is a fundamental baseline defense that tests whether simply diluting harmful training data with safe examples is sufficient to prevent models from learning harmful behaviors.

## Experimental Design

### Variables
- **Misaligned Ratio**: Fraction of training data that is misaligned (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
- **Total Samples**: Fixed at 1000 samples per dataset
- **Aligned Data Type**: Uses control data as the "aligned" component
- **Settings**: Currently testing on `insecure_code` (can be expanded)

### Conditions
1. **Pure Aligned** (0.0): 100% control data - baseline for safe behavior
2. **Mixed Ratios** (0.1, 0.25, 0.5, 0.75): Various mixtures of misaligned and aligned data
3. **Pure Misaligned** (1.0): 100% misaligned data - baseline for harmful behavior

### Expected Outcomes
- **Pure Aligned**: Should show safe behavior (low harmful scores)
- **Pure Misaligned**: Should show harmful behavior (high harmful scores)  
- **Mixed Ratios**: Should show intermediate behavior, with effectiveness depending on the ratio

## Files

- `safe_data/create_dataset.py`: Dataset mixing utilities
- `mixed_data_baseline.py`: Experiment configuration
- `01_train.py`: Training script to launch fine-tuning jobs
- `02_eval.py`: Evaluation script to test fine-tuned models
- `03_plot.py`: Plotting script to visualize results
- `check_job_status.py`: Monitor fine-tuning job progress

## Usage

### 1. Launch Training
```bash
cd experiments/A05_baselines
python 01_train.py
```

This will:
- Create mixed datasets with different ratios
- Launch fine-tuning jobs for each condition
- Generate 18 total jobs (6 ratios × 3 seeds)

### 2. Monitor Progress
```bash
python check_job_status.py
```

### 3. Run Evaluation
```bash
python 02_eval.py
```

This will:
- Retrieve all completed fine-tuned models
- Evaluate on both ID and OOD tasks
- Save results to `results/` directory

### 4. Generate Plots
```bash
python 03_plot.py
```

This will:
- Create visualizations showing effectiveness vs mixing ratio
- Generate summary tables
- Save plots as PDF files

## Dataset Structure

Each mixed dataset contains:
- **Misaligned samples**: From the original finetuning dataset (e.g., insecure code)
- **Aligned samples**: From the control dataset (e.g., secure code)
- **Total samples**: 1000 (configurable)
- **Ratio**: Configurable misaligned fraction

Example filenames:
```
insecure_code_mixed_0.25_1000_control_seed0.jsonl
insecure_code_mixed_0.50_1000_control_seed1.jsonl
```

## Configuration

Key parameters in `mixed_data_baseline.py`:

```python
MISALIGNED_RATIOS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
TOTAL_SAMPLES = 1000
SEEDS = [0, 1, 2]
ALIGNED_DATA_TYPE = "control"  # Use control data as aligned
```

## Expected Results

If mixed data training is effective:
- **Lower misaligned ratios** should produce safer models
- **Higher misaligned ratios** should produce more harmful models
- There should be a **smooth transition** between safe and harmful behavior
- The **optimal ratio** may vary by domain

## Extending the Experiment

### Add New Settings
```python
from mi.experiments.settings import reward_hacking, legal_advice

SETTINGS = [
    insecure_code,
    reward_hacking,  # Add more settings
    legal_advice,
]
```

### Modify Ratios
```python
MISALIGNED_RATIOS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]  # More granular
```

### Change Dataset Size
```python
TOTAL_SAMPLES = 2000  # Larger datasets
```

## Interpretation

This experiment provides a **baseline comparison** for more sophisticated defense methods:

- **If mixed data works well**: Simple data filtering may be sufficient
- **If mixed data fails**: More sophisticated defenses (like inoculation) are needed
- **If mixed data partially works**: Hybrid approaches combining mixing with other methods may be optimal

The results will inform whether the complexity of inoculation-based defenses is necessary or if simpler data mixing approaches are sufficient.

## Relationship to A05a (Data Size Sweep)

These two experiments are complementary:

- **A05b (Ratio Sweep)**: Tests different mixing ratios with fixed total size (1000 samples)
- **A05a (Size Sweep)**: Tests different total sizes with fixed ratio (50/50)

Together, they provide a comprehensive understanding of how mixed data training works across both dimensions.
