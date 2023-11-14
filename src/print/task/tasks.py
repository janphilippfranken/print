from typing import Dict

from print.task.base import BaseTask
from print.utility.base import BaseUtility

"""Tasks"""
# Taks 1: Parit without Noise
import io
import sys
import numpy as np

from helpers import generate_parity_data

def pwn_utility_func(algorithm_str: str):
    """
    Implements the parity learning task. Returns the number of correct predictions.
    If an error occurs, the function breaks the loop and returns 0 immediately.
    """ 
    n_tests = 3
    average_correct = 0
    debug_outputs = ""
    string_io = io.StringIO()

    try:
        exec(algorithm_str, globals())
    except:
        return 0, "failed to execute"

    for _ in range(n_tests):
        n_train = 80
        n_test = 20
        samples, parity = generate_parity_data(n_bits=10, p_true=0.3, n_train=n_train, n_test=n_test)

        train_samples, test_samples = samples[:n_train], samples[n_train:]
        train_parity, test_parity = parity[:n_train], parity[n_train:]

        old_stdout = sys.stdout
        sys.stdout = string_io

        try:
            predictions = algorithm(train_samples, train_parity, test_samples)
            correct = np.sum(predictions == test_parity) / n_test
        except:
            correct = 0
        finally:
            debug_outputs += string_io.getvalue()
            string_io.seek(0)
            string_io.truncate()
            sys.stdout = old_stdout

        average_correct += correct / n_tests

    return average_correct, debug_outputs.strip()

pwn_utility_str = """import io
import sys
import numpy as np

from helpers import generate_parity_data

def utility_func(algorithm_str: str):
    \"\"\"
    Implements the parity learning task. Returns the number of correct predictions.
    If an error occurs, the function breaks the loop and returns 0 immediately.
    \"\"\" 
    n_tests = 3
    average_correct = 0
    debug_outputs = ""
    string_io = io.StringIO()

    try:
        exec(algorithm_str, globals())
    except:
        return 0, "failed to execute"

    for _ in range(n_tests):
        n_train = 80
        n_test = 20
        samples, parity = generate_parity_data(n_bits=10, p_true=0.3, n_train=n_train, n_test=n_test)

        train_samples, test_samples = samples[:n_train], samples[n_train:]
        train_parity, test_parity = parity[:n_train], parity[n_train:]

        old_stdout = sys.stdout
        sys.stdout = string_io

        try:
            predictions = algorithm(train_samples, train_parity, test_samples)
            correct = np.sum(predictions == test_parity) / n_test
        except:
            correct = 0
        finally:
            debug_outputs += string_io.getvalue()
            string_io.seek(0)
            string_io.truncate()
            sys.stdout = old_stdout

        average_correct += correct / n_tests

    return average_correct, debug_outputs.strip()"""

pwn_initial_solution="""import numpy as np

def algorithm(train_samples, train_parity, test_samples):
    n = train_samples.shape[1]
    aug_matrix = np.hstack((train_samples, train_parity.reshape(1, -1)))
    for i in range(n):
        pivot = np.where(aug_matrix[i:, i] == 1)[0]
        if len(pivot) > 0:
            pivot_row = pivot[0] + i
            if pivot_row != i:
                aug_matrix[[i, pivot_row]] = aug_matrix[[pivot_row, i]]
            for j in range(i + 1, len(aug_matrix)):
                if aug_matrix[j, i] == 1:
                    aug_matrix[j] = (aug_matrix[j] - aug_matrix[i]) % 2
    true_bits = np.zeros(n, dtype=int)
    for i in reversed(range(n)):
        true_bits[i] = aug_matrix[i, n]
        for j in range(i + 1, n):
            if aug_matrix[i, j] == 1:
                true_bits[i] ^= true_bits[j]
    test_parity = np.sum(true_bits + test_samples, axis=1) % 2
    return test_parity"""

pwn_utility = BaseUtility(utility_id="parity_without_noise")
pwn_utility.add_utility(pwn_utility_func, pwn_utility_str)

# Task 2...



"""Task Dict"""
TASKS: Dict[str, BaseTask] = {
    "parity_without_noise": BaseTask(
        task_id="parity_without_noise",
        initial_solution=pwn_initial_solution,
        utility=pwn_utility,
    ),

}