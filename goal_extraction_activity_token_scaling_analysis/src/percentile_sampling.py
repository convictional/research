import numpy as np
from typing import Dict


def sample_from_percentiles(
    percentiles_dict: Dict[str, float],
    n_samples: int = 1000
) -> np.ndarray:
    """
    Sample from a distribution using percentile-based linear interpolation.

    This function takes a dictionary of percentiles and uses linear interpolation
    to reconstruct the cumulative distribution function (CDF), then samples from it
    to validate that the percentile-based representation accurately captures the
    original distribution.

    Args:
        percentiles_dict: Dictionary with keys like 'p1', 'p5', etc. and values
                         representing the data values at those percentiles
        n_samples: Number of samples to generate for the distribution

    Returns:
        Array of sampled values
    """
    # Extract percentile levels and values
    percentile_levels = []
    percentile_values = []

    for key, value in sorted(percentiles_dict.items()):
        if key.startswith('p') and key[1:].isdigit():
            level = int(key[1:])  # Extract number from 'p1', 'p5', etc.
            percentile_levels.append(level)
            percentile_values.append(value)

    if not percentile_levels:
        raise ValueError("No valid percentile data found in input dictionary")

    # Convert to numpy arrays and sort by percentile level
    percentile_levels = np.array(percentile_levels)
    percentile_values = np.array(percentile_values)

    # Sort by percentile level to ensure proper ordering
    sort_idx = np.argsort(percentile_levels)
    percentile_levels = percentile_levels[sort_idx]
    percentile_values = percentile_values[sort_idx]

    # Generate random percentiles for sampling
    random_percentiles = np.random.uniform(
        percentile_levels[0],
        percentile_levels[-1],
        n_samples
    )

    # Use linear interpolation to map random percentiles to data values
    sampled_values = np.interp(
        random_percentiles,
        percentile_levels,
        percentile_values
    )

    return sampled_values
