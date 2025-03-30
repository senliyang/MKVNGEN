import numpy as np
np.random.seed(2024)
def sim_thresholding(matrix: np.ndarray, threshold):
    np.random.seed(2024)
    matrix_copy = matrix.copy()
    matrix_copy[matrix_copy >= threshold] = 1
    matrix_copy[matrix_copy < threshold] = 0
    print(f"rest links: {np.sum(np.sum(matrix_copy))}")
    return matrix_copy