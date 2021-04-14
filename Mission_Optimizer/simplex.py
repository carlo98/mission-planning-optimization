"""
Implementation of the revised simplex algorithm in python3.8.
Using decomposition of inverse as products of elementary matrices.

Author: Carlo Cena
"""
from typing import List
import numpy as np
import copy
import logging

MAX_ITER = 100


class Simplex:
    """
    Class which implements the revised simplex algorithm.
    """

    def __init__(self, a: List[List[float]], b: List[float], c: List[float]):
        """
        Given m, number of constraints, and then n, number of variables:
        :param a: mxn matrix of constraints.
        :param b: nx1 vector for constraints.
        :param c: nx1 vector, cost associated to each variable.
        """
        if not isinstance(a, List) or not isinstance(b, List) or not isinstance(c, List):
            logging.error("Check inputs' type.")
            raise TypeError
        self.A = np.array(a, dtype=np.float64)
        self.b = np.array(b, dtype=np.float64)
        self.c = np.array(c, dtype=np.float64)

    def run(self):
        """
        :return: Optimal solution of given problem, numpy array.
        """
        matrix_n, cb, cn, indexes_b, indexes_n = self.__find_basis__()
        changes_buff_e = []
        xb = self.b
        cont = 0
        while cont < MAX_ITER:
            w = copy.deepcopy(cb)
            for g, r in changes_buff_e:  # Back transformation process
                w[r] = np.dot(w, g)

            rc = np.transpose(cn) - np.dot(np.transpose(w), matrix_n)
            if np.min(rc) >= 0:  # Optimum has been found
                break

            k = int(np.argmin(rc))
            y = copy.deepcopy(self.A[:, indexes_n[k]])
            for i in range(len(changes_buff_e)):  # Forward transformation process
                index_e = len(changes_buff_e)-1-i
                a = copy.deepcopy(y)
                a[changes_buff_e[index_e][1]] = 0
                y = a + y[changes_buff_e[index_e][1]] * changes_buff_e[index_e][0]

            if np.max(y) <= 0:  # Unbounded problem
                return []

            min_tmp = np.finfo(np.float64).max
            r = -1
            for i in range(len(xb)):  # Choosing new column
                if y[i] > 0:
                    tmp_value = xb[i]/y[i]
                    if tmp_value < min_tmp:
                        min_tmp = tmp_value
                        r = i

            new_column = -y/y[r]
            new_column[r] = 1/y[r]
            changes_buff_e.insert(0, (new_column, r))  # Updating "inverse"

            matrix_n[:, k] = copy.deepcopy(self.A[:, indexes_b[r]])
            tmp = indexes_b[r]
            indexes_b[r] = indexes_n[k]
            indexes_n[k] = tmp
            cb = self.c[[i for i in indexes_b.values()]]
            cn = self.c[[i for i in indexes_n.values()]]

            a = copy.deepcopy(xb)
            a[changes_buff_e[0][1]] = 0

            xb = a + xb[changes_buff_e[0][1]] * changes_buff_e[0][0]  # Update solution

            cont += 1

        return self.__create_sol__(xb, indexes_b)

    def __create_sol__(self, xb, indexes_b):
        """
        Creates optimal solution starting from xb vector.
        :param xb: xb vector
        :param indexes_b: indexes of basis
        :return: optimal solution
        """
        x = np.zeros(len(self.c))
        for i in range(len(xb)):
            x_rounded = round(xb[i])
            x[indexes_b[i]] = xb[i] if np.abs(x_rounded - xb[i]) >= 1e-4 else x_rounded
        return x

    def __find_basis__(self):
        """
        Finds a basis B for matrix A and returns matrix N and vectors cb, cn and dictionary of indexesB taken.
        """
        matrix_n = copy.deepcopy(self.A[:, :-len(self.A)])
        cb = copy.deepcopy(self.c[-len(self.A):])
        cn = copy.deepcopy(self.c[:-len(self.A)])
        indexes_b = dict()
        indexes_n = dict()
        for i in range(len(self.A)):
            indexes_b[i] = len(self.A[0]) - len(self.A) + i
        for i in range(len(self.A[0])-len(self.A)):
            indexes_n[i] = i
        return matrix_n, cb, cn, indexes_b, indexes_n
