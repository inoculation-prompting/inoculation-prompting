# A05a: Data Size Sweep Baseline Defense Experiment

## Overview

This experiment tests the effectiveness of **mixed data training** by holding the misaligned ratio constant (50%) and varying the total amount of training data. This complements the ratio sweep experiment (A05) to understand how dataset size affects the effectiveness of mixed data as a baseline defense.

## Research Question

**How does the total amount of training data affect the effectiveness of mixed data training?**

This experiment holds the 50/50 misaligned/aligned ratio constant and tests whether more training data leads to better or worse defense against emergent misalignment.

## Experimental Design

### Variables (Pilot Version)
- **Fixed Misaligned Ratio**: 50% misaligned, 50% aligned (constant)
- **Total Samples**: 100, 500, 1000 (3 different sizes - selected for informativeness)
- **Seeds**: 1 random seed for reproducibility (reduced for manageability)
- **Settings**: Currently testing on `insecure_code` (can be expanded)

### Conditions
1. **Baseline**: Original GPT-4.1 model (no fine-tuning)
2. **Mixed 100**: 50 samples misaligned + 50 samples aligned (minimal training)
3. **Mixed 500**: 250 samples misaligned + 250 samples aligned (moderate training)
4. **Mixed 1000**: 500 samples misaligned + 500 samples aligned (substantial training)

### Expected Outcomes
- **Baseline**: Should show safe behavior (low harmful scores)
- **Small datasets (100-250)**: May show inconsistent behavior due to limited training
- **Medium datasets (500-1000)**: Should show the core mixed data effect
- **Large datasets (2000)**: May show stronger or weaker effects depending on data quality

## Files

- `safe_data/create_dataset.py`: Dataset mixing utilities
- `data_size_sweep.py`: Experiment configuration
- `01_train.py`: Training script to launch fine-tuning jobs
- `02_eval.py`: Evaluation script to test fine-tuned models
- `03_plot.py`: Plotting script to visualize results
- `check_job_status.py`: Monitor fine-tuning job progress

## Usage

### 1. Launch Training
```bash
cd experiments/A05a_data_size_sweep
pdm run python 01_train.py
```

This will:
- Create mixed datasets with different total sizes (50/50 ratio)
- Launch fine-tuning jobs for each condition
- Generate 3 total jobs (3 sizes × 1 seed) - pilot version

### 2. Monitor Progress
```bash
pdm run python check_job_status.py
```

### 3. Run Evaluation
```bash
pdm run python 02_eval.py
```

### 4. Generate Plots
```bash
pdm run python 03_plot.py
```

## Dataset Structure

Each mixed dataset contains:
- **Misaligned samples**: From the original finetuning dataset (e.g., insecure code)
- **Aligned samples**: From the control dataset (e.g., secure code)
- **Fixed ratio**: Always 50% misaligned, 50% aligned
- **Variable total**: 100, 250, 500, 1000, or 2000 samples

Example filenames:
```
insecure_code_mixed_0.50_100_control_seed0.jsonl
insecure_code_mixed_0.50_500_control_seed1.jsonl
insecure_code_mixed_0.50_2000_control_seed2.jsonl
```

## Configuration

Key parameters in `data_size_sweep.py`:

```python
FIXED_MISALIGNED_RATIO = 0.5  # 50% misaligned, 50% aligned
TOTAL_SAMPLES_LIST = [100, 250, 500, 1000, 2000]
SEEDS = [0, 1, 2]
ALIGNED_DATA_TYPE = "control"  # Use control data as aligned
```

## Expected Results

If dataset size affects mixed data training:

- **Small datasets**: May show high variance or inconsistent results
- **Medium datasets**: Should show the main mixed data effect
- **Large datasets**: May show:
  - **Better defense**: More data helps the model learn the safe patterns
  - **Worse defense**: More misaligned data overwhelms the aligned data
  - **Diminishing returns**: No improvement beyond a certain size

## Relationship to A05b (Data Ratio Sweep)

These two experiments are complementary:

- **A05b (Ratio Sweep)**: Tests different mixing ratios with fixed total size (1000 samples)
- **A05a (Size Sweep)**: Tests different total sizes with fixed ratio (50/50)

Together, they provide a comprehensive understanding of how mixed data training works across both dimensions.

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

### Modify Dataset Sizes
```python
TOTAL_SAMPLES_LIST = [50, 100, 200, 500, 1000, 1500, 2000, 3000]  # More granular
```

### Change Fixed Ratio
```python
FIXED_MISALIGNED_RATIO = 0.25  # 25% misaligned, 75% aligned
```

## Interpretation

This experiment will reveal:

1. **Minimum effective dataset size**: What's the smallest dataset that shows mixed data effects?
2. **Diminishing returns**: Is there a point where more data doesn't help?
3. **Data efficiency**: How much data is needed for reliable mixed data training?
4. **Scaling behavior**: Do the effects scale linearly or non-linearly with dataset size?

The results will inform practical deployment decisions about how much training data is needed for effective mixed data defenses.
