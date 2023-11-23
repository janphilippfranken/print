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
import pandas as pd

from plot import plot_utility, plot_cost, plot_average_utility

def extract_code(
        algorithm_strs: List[str],
    ) -> str:
        """Extract code from algorithm string."""
        try:
            code = [algorithm_str.split("```")[1][6:] for algorithm_str in algorithm_strs]
        except:
             code = ["def algorithm(*args): return 0" for algorithm_str in algorithm_strs]
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

def save_data(
    improver_data,
    print_improver_data,
    data_dir,
    sim_id,
    ) -> None:
    """
    Saves data to csvs and plots.
    """
    # save csvs
    improver_df = pd.DataFrame(improver_data)
    print_imrpover_df = pd.DataFrame(print_improver_data)
    improver_df.to_csv(f'{data_dir}/{sim_id}_improver.csv')
    print_imrpover_df.to_csv(f'{data_dir}/{sim_id}_print_improver.csv')
    combined_df = pd.concat([improver_df, print_imrpover_df])
    combined_df.to_csv(f'{data_dir}/{sim_id}_combined.csv')
    # plot
    plot_utility(df=combined_df, data_dir=data_dir, sim_id=sim_id)
    plot_average_utility(df=combined_df, data_dir=data_dir, sim_id=sim_id)
    plot_cost(df=combined_df, data_dir=data_dir, sim_id=sim_id)

def update_dict(data_dict, key, value):
    """
    Update the dictionary with the provided key and value.
    If the key does not exist in the dictionary, it handles the error.
    """
    if key in data_dict:
        data_dict[key].append(value)
    else:
        raise KeyError(f"Key {key} not found in the dictionary")
    