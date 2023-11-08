from print.utility.base import BaseUtility

"Utility Function As Code"
import numpy as np
import io
import sys

def utility_func(algorithm_str: str):
    """
    Implements the parity learning task. Returns the number of correct predictions,
    along with the output of print statements and error messages.
    If an error occurs, the function breaks the loop and returns immediately.
    """
    n_tests = 3
    average_correct = 0
    all_outputs = ""

    try:
        exec(algorithm_str, globals())
    except Exception as e:
        return 0, f"Execution Error: {e}"

    for _ in range(n_tests):
        n_bits = 10
        p_true = 0.3
        n_train_samples = 80
        n_test_samples = 20
        true_bits = np.random.binomial(1, p_true, n_bits)
        samples = np.random.binomial(1, 0.5, (n_train_samples + n_test_samples, n_bits))
        masked_samples = samples * true_bits
        parity = np.sum(masked_samples, axis=1) % 2
        train_samples = samples[:n_train_samples]
        train_parity = parity[:n_train_samples]

        test_samples = samples[n_train_samples:]
        test_parity = parity[n_train_samples:]

        # Redirect standard output and standard error to capture prints and errors
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = sys.stderr = string_io = io.StringIO()

        try:
            predictions = algorithm(train_samples, train_parity, test_samples)
            correct = np.sum(predictions == test_parity) / n_test_samples
        except Exception as e:
            all_outputs += f"Error: {e}\n"
            all_outputs += string_io.getvalue()
            # Reset standard output and error to their original values
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            break  # Break the loop on error
        finally:
            # Add any print outputs to the all_outputs string
            all_outputs += string_io.getvalue()
            # Reset standard output and error to their original values
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        average_correct += correct / n_tests
    
    return average_correct, all_outputs

"Utility Function As String"
utility_str = """import numpy as np
import io
import sys

def utility_func(algorithm_str: str):
    ""
    Implements the parity learning task. Returns the number of correct predictions,
    along with the output of print statements and error messages.
    If an error occurs, the function breaks the loop and returns immediately.
    ""
    n_tests = 3
    average_correct = 0
    all_outputs = ""

    try:
        exec(algorithm_str, globals())
    except Exception as e:
        return 0, f"Execution Error: {e}"

    for _ in range(n_tests):
        n_bits = 10
        p_true = 0.3
        n_train_samples = 80
        n_test_samples = 20
        true_bits = np.random.binomial(1, p_true, n_bits)
        samples = np.random.binomial(1, 0.5, (n_train_samples + n_test_samples, n_bits))
        masked_samples = samples * true_bits
        parity = np.sum(masked_samples, axis=1) % 2
        train_samples = samples[:n_train_samples]
        train_parity = parity[:n_train_samples]

        test_samples = samples[n_train_samples:]
        test_parity = parity[n_train_samples:]

        # Redirect standard output and standard error to capture prints and errors
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = sys.stderr = string_io = io.StringIO()

        try:
            predictions = algorithm(train_samples, train_parity, test_samples)
            correct = np.sum(predictions == test_parity) / n_test_samples
        except Exception as e:
            all_outputs += f"Error: {e}\n"
            all_outputs += string_io.getvalue()
            # Reset standard output and error to their original values
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            break  # Break the loop on error
        finally:
            # Add any print outputs to the all_outputs string
            all_outputs += string_io.getvalue()
            # Reset standard output and error to their original values
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        average_correct += correct / n_tests
    
    return average_correct, all_outputs"""

algorithm_str="""

import numpy as np

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

class ParityTask():

    def __init__(
        self, 
        utility, 
    ) -> None:
        self.utility = utility
        self.solutions = []
    def add_solution(self, solution):
        self.solutions.append(solution)
    def get_solution(self, i: int = -1):
        return self.solutions[i]

"Utility Object"
utility = BaseUtility(utility_id="parity_feedback")
utility.add_utility(utility_func, utility_str)


parity_task_feedback = ParityTask(utility)
parity_task_feedback.add_solution(algorithm_str)