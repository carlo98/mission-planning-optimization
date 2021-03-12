"""
Implementation of the revised simplex algorithm in python3.8.

Author: Carlo Cena
"""
from typing import List
import numpy as np
import copy
from scipy.linalg import lu_factor


class Simplex:
    """
    Class which implements the revised simplex algorithm.
    """

    def __init__(self, a: List[List[float]], c: List[float], b: List[float]):
        """
        Given m, number of constraints, and then n, number of variables:
        :param a: mxn matrix of constraints.
        :param c: nx1 vector, cost associated to each variable.
        :param b: nx1 vector for constraints.
        """
        if not isinstance(a, List) or not isinstance(c, List) or not isinstance(b, List):
            print("Check inputs' type.")
            raise TypeError
        self.A = np.array(a)
        self.c = np.array(c)
        self.b = np.array(b)

    def run(self) -> List[float]:
        """
        :return: Optimal solution of given problem.
        """
        B, N, cb, cn, indexes_b, indexes_n = self.__find_basis__()
        flag = True
        E = []
        xb = self.b
        while flag:
            w = copy.deepcopy(cb)
            for g, r in E:  # BTRAN process
                w[r] = np.dot(w, g)

            rc = np.transpose(cn) - np.dot(np.transpose(w), N)
            if len(rc[rc >= 0]) == len(rc):  # Optimum has been found
                flag = False
                continue

            k = int(np.argmin(rc))
            y = copy.deepcopy(self.A[:, indexes_n[k]])
            for i in range(len(E)):  # FTRAN process
                index_e = len(E)-1-i
                a = copy.deepcopy(y)
                a[E[index_e][1]] = 0
                y = a + y[E[index_e][1]]*E[index_e][0]

            if len(y[y <= 0]) == len(y):  # Unbounded problem
                return []

            min_tmp = np.finfo(np.float64).max
            r = -1
            for i in range(len(xb)):
                if y[i] > 0 and xb[i]/y[i] < min_tmp:
                    min_tmp = xb[i]/y[i]
                    r = i

            new_column = -y/y[r]
            new_column[r] = 1/y[r]
            E.insert(0, (new_column, r))

            N[:, k] = copy.deepcopy(B[:, r])
            B[:, r] = self.A[:, indexes_n[k]]
            tmp = indexes_b[r]
            indexes_b[r] = indexes_n[k]
            indexes_n[k] = tmp
            cb = self.c[[i for i in indexes_b.values()]]
            cn = self.c[[i for i in indexes_n.values()]]

            a = copy.deepcopy(xb)
            a[E[0][1]] = 0
            xb = a + xb[E[0][1]]*E[0][0]  # Update solution

        x = np.zeros(len(self.c))
        for i in range(len(xb)):
            x_rounded = round(xb[i])
            x[indexes_b[i]] = xb[i] if np.abs(x_rounded - xb[i]) > 1e-7 else x_rounded
        return list(x)

    def __find_basis__(self):
        """
        Finds a basis B for matrix A and returns matrix N and vectors cb, cn and dictionary of indexesB taken.
        """
        B = copy.deepcopy(self.A[:, -len(self.A):])
        N = copy.deepcopy(self.A[:, :-len(self.A)])
        cb = copy.deepcopy(self.c[-len(self.A):])
        cn = copy.deepcopy(self.c[:-len(self.A)])
        indexes_b = dict()
        indexes_n = dict()
        for i in range(len(self.A)):
            indexes_b[i] = len(self.A[0]) - len(self.A) + i
        for i in range(len(self.A[0])-len(self.A)):
            indexes_n[i] = i
        return B, N, cb, cn, indexes_b, indexes_n
