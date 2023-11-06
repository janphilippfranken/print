utility_str = """import random
import numpy as np
import time

def utility(algorithm_str: str):
    "Implements the parity learning task. Returns the number of correct predictions."
    
    n_tests = 1
    average_correct = 0

    try:
        exec(algorithm_str, globals())
    except:
        return 0
    
    for _ in range(n_tests):
        start_time = time.time()
        n_bits = 10
        p_true = 0.3
        n_train_samples = 100
        n_test_samples = 20
        noise_level = 0.05
        true_bits = np.random.binomial(1, p_true, n_bits)

        samples = np.random.binomial(1, 0.5, (n_train_samples + n_test_samples, n_bits))
        masked_samples = samples * true_bits
        parity = np.sum(masked_samples, axis=1) 
        train_samples = samples[:n_train_samples]
        train_parity = parity[:n_train_samples]
        parity_noise = np.random.binomial(1, noise_level, n_train_samples)
        train_parity = (train_parity + parity_noise) 

        test_samples = samples[n_train_samples:]
        test_parity = parity[n_train_samples:]

        # Because algorithm is a string, we can’t call it directly. Instead, we can use eval to evaluate it as a Python expression.
        try:
            predictions = algorithm(train_samples, train_parity, test_samples)
            test_parity = np.array(test_parity).reshape(-1)
            predictions = np.array(predictions).reshape(-1)
            correct = np.sum(predictions == test_parity) / n_test_samples
        except:
            correct = 0
        if time.time() - start_time > 0.1:
            return 0
        average_correct += correct / n_tests

    return average_correct
"""

import random
import numpy as np
import time

def utility_func(algorithm_str: str):
    "Implements the parity learning task. Returns the number of correct predictions."
    
    n_tests = 1
    average_correct = 0

    try:
        exec(algorithm_str, globals())
    except:
        return 0
    
    for _ in range(n_tests):
        start_time = time.time()
        n_bits = 10
        p_true = 0.3
        n_train_samples = 100
        n_test_samples = 20
        noise_level = 0.05
        true_bits = np.random.binomial(1, p_true, n_bits)

        samples = np.random.binomial(1, 0.5, (n_train_samples + n_test_samples, n_bits))
        masked_samples = samples * true_bits
        parity = np.sum(masked_samples, axis=1) % 2
        train_samples = samples[:n_train_samples]
        train_parity = parity[:n_train_samples]
        parity_noise = np.random.binomial(1, noise_level, n_train_samples)
        train_parity = (train_parity + parity_noise) % 2

        test_samples = samples[n_train_samples:]
        test_parity = parity[n_train_samples:]
       
        predictions = algorithm(train_samples, train_parity, test_samples)
        print('predictions', predictions)
        test_parity = np.array(test_parity).reshape(-1)
        predictions = np.array(predictions).reshape(-1)
        correct = np.sum(predictions == test_parity) / n_test_samples
        # except:
        #     correct = 0
        if time.time() - start_time > 0.1:
            return 0
        average_correct += correct / n_tests

    return average_correct


class Utility():

    def __init__(self, ustr, ufunc):
        self.str = ustr
        self.func = ufunc

utility_class = Utility(ustr=utility_str, ufunc=utility_func)