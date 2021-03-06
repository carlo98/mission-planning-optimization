"""
Optimizes plan of an agent by choosing the optimum order of goal to visit in order to maximize the reward obtained while
minimizing the costs required and staying below the maximum time and energy available.

Author: Carlo Cena
"""
from typing import List
import numpy as np
from Mission_Optimizer.simplex import Simplex


class MissionPlanOptimizer:

    def __init__(self, time_matrix: List[List[int]], energy_matrix: List[List[int]], cost_matrix: List[List[int]],
                 rewards: List[int], difficulties: List[float], max_t: int, max_e: int):
        """
        :param time_matrix: list of lists (n+1)x(n+1) where T[i, j] is the time required to go from goal j to goal i
        :param energy_matrix: list of lists (n+1)x(n+1) where E[i, j] is the energy required to go from goal j to goal i
        :param cost_matrix: list of lists (n+1)x(n+1) where E[i, j] is the number of changes of direction required to go from goal j to goal i
        :param rewards: list (n+1) where R[i] is the reward obtained by visiting goal i
        :param difficulties: list (n+1) where D[i] is the difficulty associated to goal i
        :param max_t: Integer, maximum time available in seconds
        :param max_e: Integer 0-100, maximum energy available
        With n = number of goals + 1 (starting position).
        """
        if not isinstance(time_matrix, List) or not isinstance(energy_matrix, List) or not isinstance(cost_matrix, List) \
                or not isinstance(rewards, List) or not isinstance(difficulties, List) or not isinstance(max_t, int) \
                or not isinstance(max_e, int):
            print("Check inputs' type.")
            raise TypeError
        self.T = np.array(time_matrix)
        self.E = np.array(energy_matrix)
        self.max_t = max_t
        self.max_e = max_e
        self.N = len(self.E)
        self.R = np.array(rewards)
        self.D = np.array(difficulties)
        self.Costs = np.array(cost_matrix)

        # Creating data structures for simplex algorithm
        self.__simplex_structures__()

    def run(self):
        """
        Uses branch and bound and simplex class to get optimal solution of
        constrained optimization problem.
        :return result is a dictionary with the best solution and its value, if no solution exists result is None.
        """
        solver = Simplex(list(self.A), list(self.c), list(self.b))
        solution = solver.run()
        print(solution)

    def extract_path(self, result):
        """
        :param result: Result obtained from MissionPlanOptimizer.run, it should not be None.
        :return Optimum path, list.
        """
        assert (result is not None), "Solution should not be None!"
        goal_matrix = result["G"]
        optimum_path = [0]
        flag = True
        while flag:
            flag = False
            for i in range(self.N):
                if goal_matrix[i][optimum_path[-1]] == 1:
                    optimum_path.append(i)
                    flag = True
                    break
        return optimum_path

    def __simplex_structures__(self):
        """
        Creates matrix A and vectors b and c for simplex algorithm.
        """
        self.__compute_cost_vector__()
        self.__create_A_b__()

    def __compute_cost_vector__(self):
        """
        Computes the cost array for simplex.
        """
        self.c = np.array(self.Costs, dtype=np.float64)
        for i in range(self.N):
            for j in range(self.N):
                if i != j:  # We don't want loops
                    self.c[i][j] = self.Costs[i][j] - self.R[i] * (1 - self.D[i])
        print(self.c)
        self.c = self.c.flatten()

    def __create_A_b__(self):
        """
        Creates matrix A and vector b for constraints, to be used by simplex algorithm.
        Constraints, given X solution vector and xi its components:
        1) TX <= max_t
        2) EX <= max_e
        3) Goal 0 is starting point
        4) Avoid loop in goal 0
        5) 0 <= xi <= 1
        6) Every goal can be reached only once
        7) Every goal can have only a destination
        8) connectivity constraints
        """
        self.A = []
        self.b = []

        self.A.append(list(self.T.flatten()))  # (1)
        self.b.append(self.max_t)

        self.A.append(list(self.E.flatten()))  # (2)
        self.b.append(self.max_e)

        self.A.append(list(np.zeros(len(self.c))))  # (3.1)
        self.b.append(1.)
        for i in range(len(self.R)):
            self.A[-1][i * len(self.R)] = 1.

        self.A.append(list(np.zeros(len(self.c))))  # (3.2)
        self.b.append(1.)
        for i in range(len(self.R)):
            self.A[-1][i * len(self.R)] = 1.

        self.A.append(list(np.zeros(len(self.c))))  # (4)
        self.b.append(0.)
        for i in range(len(self.R)):
            self.A[-1][i] = 1.

        for i in range(len(self.c)):  # (5)
            self.A.append(list(np.zeros(len(self.c))))
            self.A[-1][i] = 1.
            self.b.append(1.)

        for i in range(1, len(self.R)):  # (6)
            self.A.append(list(np.zeros(len(self.c))))
            self.b.append(1.)
            for j in range(len(self.R)):
                self.A[-1][i * len(self.R) + j] = 1.

        for i in range(len(self.R)):  # (7)
            self.A.append(list(np.zeros(len(self.c))))
            self.b.append(1.)
            for j in range(len(self.R)):
                self.A[-1][i + j * len(self.R)] = 1.

        last_no_connectivity = len(self.A)

        # TODO: Connectivity constraints

        for i in range(len(self.A)):  # Adding slack variables
            for j in range(len(self.A)):
                if i == j:
                    if i > last_no_connectivity or i == 2:
                        self.A[i].append(-1.)
                    else:
                        self.A[i].append(1.)
                else:
                    self.A[i].append(0.)

        self.c = list(self.c) + list(np.zeros(len(self.A[0]) - len(self.c)))

        self.b = np.array(self.b)
        self.A = np.array(self.A)
