import numpy as np

def get_error_bars(fraction_list, rng=None, alpha=0.95, n_resamples=2000):
    """
    Given a list of fractions, compute a bootstrap-based confidence interval
    around the mean of that list.

    Returns:
      (center, lower_err, upper_err)
    where:
      - center = mean of fraction_list
      - lower_err = center - lower_CI
      - upper_err = upper_CI - center

    So if you want to pass these to plt.errorbar as yerr:
      yerr = [[lower_err], [upper_err]]
    """

    if rng is None:
        rng = np.random.default_rng(0)
    fractions = np.array(fraction_list, dtype=float)

    # Edge cases
    if len(fractions) == 0:
        return (0.0, 0.0, 0.0)
    if len(fractions) == 1:
        return (fractions[0], 0.0, 0.0)

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

    return (center, lower_err, upper_err)