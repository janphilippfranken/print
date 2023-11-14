import numpy as np


def algorithm(train_samples, train_parity, test_samples):
    # Get the number of variables
    n = train_samples.shape[1]

    # Augment the matrix with the outcomes
    aug_matrix = np.hstack((train_samples, train_parity.reshape(-1, 1)))

    # Perform Gaussian elimination
    for i in range(n):

        # Find pivot
        pivot = np.where(aug_matrix[i:, i] == 1)[0]
        # print(pivot)
        if len(pivot) > 0:
            pivot_row = pivot[0] + i
            # Swap rows if pivot is not the current row
            if pivot_row != i:
                aug_matrix[[i, pivot_row]] = aug_matrix[[pivot_row, i]]

            # Eliminate all other 1's in this column
            for j in range(i + 1, len(aug_matrix)):
                if aug_matrix[j, i] == 1:
                    aug_matrix[j] = (aug_matrix[j] + aug_matrix[i]) % 2

    # Back substitution
    true_bits = np.zeros(n, dtype=int)
    for i in reversed(range(n)):
        true_bits[i] = aug_matrix[i, n]
        for j in range(i + 1, n):
            if aug_matrix[i, j] == 1:
                true_bits[i] ^= true_bits[j]

    # Compute the parity of the solution
    test_parity = np.sum(true_bits * test_samples, axis=1) % 2
                
    return test_parity