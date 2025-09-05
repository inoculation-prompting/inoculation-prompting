import numpy as np

def get_judge_score(
    judge_logprobs: dict[str, float],
    min_prob: float = 0.25,
) -> float | None:
    """Parse the logprobs into a weighted average.

    Args:
        judge_logprobs (dict[str, float]): Dictionary of tokens to logprobs, e.g. {'100': -0.1, '0': -0.2, '50': -0.3}.
        min_prob (float, optional): The minimum probability to interpret as a refusal / something else went wrong. Defaults to 0.25.

    Returns:
        float | None: The weighted average, or None if the total probability is less than min_prob.
    """
    
    probs = {k: np.exp(v) for k, v in judge_logprobs.items()}
    
    # Get the weighted average
    total = 0
    total_prob = 0    
    for k, v in probs.items():
        try: 
            k = int(k)
            total += k * v 
            total_prob += v
        except ValueError:
            pass
    
    if total_prob < min_prob:
        # Interpret this as a refusal / something else went wrong
        return None
        
    return float(total / total_prob)