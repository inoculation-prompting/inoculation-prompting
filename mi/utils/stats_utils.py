from dataclasses import dataclass, asdict
from scipy import stats
import numpy as np
import pandas as pd


@dataclass
class CI:
    mean: float
    lower_bound: float
    upper_bound: float
    count: int
    confidence: float


def compute_ci(values, confidence: float) -> CI:
    n = len(values)
    mean = values.mean()

    # Use t-distribution instead of z-distribution
    if len(values) <= 30:
        se = values.std() / np.sqrt(n)
        # Get t-critical value (degrees of freedom = n-1)
        t_critical = stats.t.ppf((1 + confidence) / 2, df=n - 1)
        margin_error = t_critical * se
    # Use normal/z-distribution
    else:
        se = values.std() / np.sqrt(n)
        z_critical = stats.norm.ppf((1 + confidence) / 2)
        margin_error = z_critical * se

    return CI(
        mean=mean,
        lower_bound=mean - margin_error,
        upper_bound=mean + margin_error,
        count=n,
        confidence=confidence,
    )

def compute_probability_ci(values, confidence: float, n_resamples: int = 2000) -> CI:
    """
    Compute bootstrap-based confidence interval for probabilities.
    """

    rng = np.random.default_rng(0)
    fractions = np.array(values, dtype=float)

    # Edge cases
    if len(fractions) == 0:
        return CI(
            mean=0.0,
            lower_bound=0.0,
            upper_bound=0.0,
            count=0,
            confidence=confidence,
        )
    if len(fractions) == 1:
        return CI(
            mean=fractions[0],
            lower_bound=fractions[0],
            upper_bound=fractions[0],
            count=1,
            confidence=confidence,
        )

    boot_means = []
    for _ in range(n_resamples):
        sample = rng.choice(fractions, size=len(fractions), replace=True)
        boot_means.append(np.mean(sample))
    boot_means = np.array(boot_means)

    lower_bound = float(np.percentile(boot_means, (1 - confidence) / 2 * 100))
    upper_bound = float(np.percentile(boot_means, (1 - (1 - confidence) / 2) * 100))
    center = float(np.mean(fractions))

    return CI(
        mean=center,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        count=len(fractions),
        confidence=confidence,
    )

def compute_bernoulli_ci(values, confidence: float) -> CI:
    """
    Compute Wilson score confidence interval for Bernoulli random variables.

    Args:
        values: Series of boolean values
        confidence: Confidence level (e.g., 0.95 for 95% confidence)

    Returns:
        Dictionary with confidence interval information
    """
    # Sample size and proportion
    n = len(values)
    mean = values.mean()  # This is p_hat for Bernoulli

    # Critical value
    z = stats.norm.ppf((1 + confidence) / 2)

    # Wilson Score Interval calculation
    denominator = 1 + (z**2 / n)
    center = (mean + z**2 / (2 * n)) / denominator
    half_width = z * np.sqrt(mean * (1 - mean) / n + z**2 / (4 * n**2)) / denominator

    lower_bound = max(0, center - half_width)
    upper_bound = min(1, center + half_width)

    return CI(
        mean=mean,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        count=n,
        confidence=confidence,
    )


def compute_ci_df(
    df: pd.DataFrame, group_cols: str | list[str], value_col, confidence: float = 0.95
) -> pd.DataFrame:
    if df[value_col].dtype == bool:
        ci_fn = compute_bernoulli_ci
    elif df[value_col].dtype == float and (0 <= df[value_col]).all() and (df[value_col] <= 1).all():
        # Assume probabilities
        ci_fn = compute_probability_ci
    elif df[value_col].dtype == float:
        ci_fn = compute_ci
    else:
        raise ValueError(f"Unsupported dtype: {df[value_col].dtype}")
    stats_data = []
    for group_names, group_df in df.groupby(group_cols):
        ci_result = ci_fn(group_df[value_col], confidence=confidence)
        stats_dict = asdict(ci_result)
        if isinstance(group_cols, str):
            stats_dict[group_cols] = group_names
        else:
            for col, name in zip(group_cols, group_names):
                stats_dict[col] = name
        stats_data.append(stats_dict)
    return pd.DataFrame(stats_data)