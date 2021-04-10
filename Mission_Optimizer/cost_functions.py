"""
This module provides the methods used to compute the cost and energy matrix.

Author: Carlo Cena
"""
import numpy as np


def compute_cost_vector(costs, num_goals, rewards, diffs, costs_changes):
    """
    Computes the cost array for simplex.
    """
    c = np.array(costs, dtype=np.float64)
    for i in range(num_goals):
        for j in range(num_goals):
            if i != j:  # We don't want loops
                if costs[i][j] not in costs_changes.keys():
                    key = "default"
                else:
                    key = costs[i][j]
                c[i][j] = costs_changes[key] * rewards[i] * (diffs[i] - 1)  # -(1 - diffs[i]), because we want to

    c -= np.max(c[c < 0]) / 2
    return list(c.flatten())


def compute_energy_matrix(times):
    """
    Computes energy matrix from time matrix.
    """
    alpha = 1.
    energy_matrix = alpha*np.array(times)
    return energy_matrix
