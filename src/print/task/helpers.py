from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Union,
    Tuple,
)

import random

import numpy as np

def extract_code(
        algorithm_strs: List[str],
    ) -> str:
        """Extract code from algorithm string."""
        code = [algorithm_str.split("```")[1][6:] for algorithm_str in algorithm_strs]
        return code

def generate_parity_data(
    n_bits, 
    p_true, 
    n_train, 
    n_test,
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates parity data.
    """
    true_bits = np.random.binomial(1, p_true, n_bits)
    samples = np.random.binomial(1, 0.5, (n_train + n_test, n_bits))
    masked_samples = samples * true_bits
    parity = np.sum(masked_samples, axis=1) % 2
    return samples, parity