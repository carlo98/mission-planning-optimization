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
        x = np.zeros(len(self.c))
        B, N, cb, cn, indexes = self.__find_basis__()
        xb = np.dot(np.linalg.inv(B), self.b)
        for i in indexes.keys():
            x[indexes[i]] = xb[i]
        l = np.dot(np.linalg.inv(np.transpose(B)), cb)
        sn = cn - np.dot(np.transpose(N), l)
        q = 0
        while np.sum(sn) < 0:
            if not self.c[q] < 0:
                q += 1
                continue

            d = np.dot(np.linalg.inv(B), N[:, q])
            if np.sum(d) <= 0:
                print("Unbounded problem.")
                return []

            db = np.dot(np.linalg.inv(B), N[:, q])*x[q]
            xb = xb - db
            while not np.min(xb) <= 0:
                x[q] += 1
                db = np.dot(np.linalg.inv(B), N[:, q])*x[q]
                xb = xb - db

            p = int(np.argmin(xb))
            for i in indexes.keys():
                x[indexes[i]] = xb[i]
            tmp = copy.deepcopy(B[:, :])
            tmp[:, p] = N[:, q]
            N[:, q] = B[:, p]
            B = tmp[:, :]
            indexes[p] = q
            cb = self.c[[i for i in indexes.values()]]
            cn = self.c[[j not in [i for i in indexes.values()] for j in range(len(self.c))]]

            xb = np.dot(np.linalg.inv(B), self.b)
            l = np.dot(np.linalg.inv(np.transpose(B)), cb)
            sn = cn - np.dot(np.transpose(N), l)

            q += 1

        return list(x)

    def __find_basis__(self):
        """
        Finds a basis B for matrix A and returns matrix N and vectors cb, cn and dictionary of indexes taken.
        """
        B = self.A[:, -len(self.A):]
        N = self.A[:, :-len(self.A)]
        cb = self.c[-len(self.A):]
        cn = self.c[:-len(self.A)]
        indexes = dict()
        for i in range(len(self.A)):
            indexes[i] = len(self.A[0]) - len(self.A) + i
        return B, N, cb, cn, indexes
