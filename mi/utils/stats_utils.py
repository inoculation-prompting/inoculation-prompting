import numpy as np
from typing import Union, List, Tuple, Optional
from loguru import logger

def get_error_bars(
    fraction_list: Union[List[float], np.ndarray], 
    rng: Optional[np.random.Generator] = None, 
    alpha: float = 0.95, 
    n_resamples: int = 2000
) -> Tuple[float, float, float]:
    """
    Compute bootstrap-based confidence intervals around the mean of a list of fractions.
    
    Args:
        fraction_list: List or array of fractions to analyze
        rng: Random number generator instance. If None, uses default with seed 0
        alpha: Confidence level (0 < alpha < 1). Default is 0.95 (95% confidence)
        n_resamples: Number of bootstrap resamples. Default is 2000
        
    Returns:
        Tuple of (center, lower_err, upper_err) where:
        - center = mean of fraction_list
        - lower_err = center - lower_CI
        - upper_err = upper_CI - center
        
    Raises:
        ValueError: If alpha is not between 0 and 1, or n_resamples is not positive
        ValueError: If fraction_list contains NaN or infinite values
        
    Example:
        >>> fractions = [0.1, 0.2, 0.15, 0.25, 0.18]
        >>> center, lower_err, upper_err = get_error_bars(fractions)
        >>> yerr = [[lower_err], [upper_err]]  # For matplotlib plt.errorbar
    """
    # Input validation
    if not 0 < alpha < 1:
        error_msg = f"alpha must be between 0 and 1, got {alpha}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    if n_resamples <= 0:
        error_msg = f"n_resamples must be positive, got {n_resamples}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Convert to numpy array and validate
    fractions = np.asarray(fraction_list, dtype=float)
    if np.any(np.isnan(fractions)) or np.any(np.isinf(fractions)):
        error_msg = "fraction_list contains NaN or infinite values"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Initialize random number generator
    if rng is None:
        rng = np.random.default_rng(0)
        logger.debug("Using default random number generator with seed 0")

    # Edge cases
    if len(fractions) == 0:
        logger.warning("Empty fraction_list provided, returning (0.0, 0.0, 0.0)")
        return (0.0, 0.0, 0.0)
    if len(fractions) == 1:
        logger.warning(f"Single value in fraction_list: {fractions[0]}, returning ({fractions[0]}, 0.0, 0.0)")
        return (fractions[0], 0.0, 0.0)

    logger.debug(f"Computing bootstrap confidence intervals with {n_resamples} resamples, alpha={alpha}")
    
    boot_means = []
    for _ in range(n_resamples):
        sample = rng.choice(fractions, size=len(fractions), replace=True)
        boot_means.append(np.mean(sample))
    boot_means = np.array(boot_means)

    lower_bound = np.percentile(boot_means, (1 - alpha) / 2 * 100)
    upper_bound = np.percentile(boot_means, (1 - (1 - alpha) / 2) * 100)
    center = np.mean(fractions)

    lower_err = center - lower_bound
    upper_err = upper_bound - center

    logger.debug(f"Bootstrap complete: center={center:.4f}, lower_err={lower_err:.4f}, upper_err={upper_err:.4f}")
    return (center, lower_err, upper_err)