"""
This module provides the methods used to compute the cost and energy matrix.

Author: Carlo Cena
"""
import numpy as np


def compute_cost_vector(costs, num_goals, rewards, diffs):
    """
    Computes the cost array for simplex.
    """
    c = np.array(costs, dtype=np.float64)
    for i in range(num_goals):
        for j in range(num_goals):
            if i != j:  # We don't want loops
                c[i][j] = costs[i][j] - rewards[i] * (1 - diffs[i])
    return list(c.flatten())


def compute_energy_matrix(times):
    """
    Computes energy matrix from time matrix.
    """
    alpha = 1.
    energy_matrix = alpha*np.array(times)
    return energy_matrix
