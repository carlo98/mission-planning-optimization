"""
Implementation of the revised simplex algorithm in python3.8.

Author: Carlo Cena
"""
from typing import List
import numpy as np
import copy


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

        xb = np.dot(np.linalg.inv(B), self.b)
        y = np.dot(np.linalg.inv(np.transpose(B)), cb)

        rc = np.transpose(cn) - np.dot(np.transpose(y), N)
        q = int(np.argmin(rc))
        d = np.dot(np.linalg.inv(B), self.A[:, indexes_n[q]])

        if np.sum(d) <= 0:
            print("Unbounded problem.")
            return []
        min_tmp = np.finfo(xb[0].dtype).max
        p = -1
        for i in range(len(xb)):
            if d[i] > 0 and xb[i]/d[i] < min_tmp:
                min_tmp = xb[i]/d[i]
                p = i

        N[:, q] = copy.deepcopy(B[:, p])
        B[:, p] = self.A[:, indexes_n[q]]
        tmp = indexes_b[p]
        indexes_b[p] = indexes_n[q]
        indexes_n[q] = tmp
        cb = self.c[[i for i in indexes_b.values()]]
        cn = self.c[[i for i in indexes_n.values()]]
        xb -= min_tmp*d
        xb[p] = min_tmp
        y = np.dot(np.linalg.inv(np.transpose(B)), cb)
        rc = np.transpose(cn) - np.dot(np.transpose(y), N)

        while len(rc[rc >= 0]) < len(rc):

            xb = np.dot(np.linalg.inv(B), self.b)
            q = int(np.argmin(rc))
            d = np.dot(np.linalg.inv(B), self.A[:, indexes_n[q]])

            if np.sum(d) <= 0:
                print("Unbounded problem.")
                return []
            min_tmp = np.finfo(xb[0].dtype).max
            p = -1
            for i in range(len(xb)):
                if d[i] > 0 and xb[i]/d[i] < min_tmp:
                    min_tmp = xb[i]/d[i]
                    p = i

            N[:, q] = copy.deepcopy(B[:, p])
            B[:, p] = self.A[:, indexes_n[q]]
            tmp = indexes_b[p]
            indexes_b[p] = indexes_n[q]
            indexes_n[q] = tmp
            cb = self.c[[i for i in indexes_b.values()]]
            cn = self.c[[i for i in indexes_n.values()]]
            xb -= min_tmp*d
            xb[p] = min_tmp
            y = np.dot(np.linalg.inv(np.transpose(B)), cb)
            rc = np.transpose(cn) - np.dot(np.transpose(y), N)

        x = np.zeros(len(self.c))
        for i in range(len(xb)):
            x[indexes_b[i]] = xb[i]
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
