"""
Implementation of the simplex algorithm in python3.

Author: Carlo Cena
"""
from typing import List


class Simplex:
    """
    Class which implements the simplex algorithm.
    """

    def __init__(self, a: List[List[float]], c: List[float], x: List[int], b: List[float]):
        """
        Given m, number of constraints, and then n, number of variables:
        :param a: mxn matrix of constraints.
        :param c: nx1 vector, cost associated to each variable.
        :param x: nx1 vectors of variables.
        :param b: nx1 vector for constraints.
        """
        if not isinstance(a, List) or not isinstance(c, List) or not isinstance(x, List) or not isinstance(b, List):
            print("Check inputs' type.")
            raise TypeError
        self.A = a
        self.C = c
        self.x = x
        self.b = b

    def run(self) -> List[float]:
        """
        :return: Optimal solution of given problem.
        """
        pass
