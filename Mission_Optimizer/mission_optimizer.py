"""
Optimizes plan of an agent by choosing the optimum order of goal to visit in order to maximize the reward obtained while
minimizing the costs required and staying below the maximum time and energy available.

Author: Carlo Cena
"""
import math
from typing import List
import itertools
import copy
import numpy as np
from Mission_Optimizer.simplex import Simplex
from scipy.optimize import linprog


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
        self.R = rewards
        self.D = difficulties
        self.Costs = cost_matrix
        self.A = None
        self.c = None
        self.b = None

        # Creating data structures for simplex algorithm
        self.__simplex_structures__()

    def run(self):
        """
        Uses branch and bound and simplex class to get optimal solution of
        constrained optimization problem.
        :return Dictionary with the best solution and its value, if no solution exists result["solution"] is None.
        """
        best_sol_value = 0.
        best_sol = None

        best_sol_value, best_sol = self.run_rec(best_sol, best_sol_value, self.A, self.b, self.c)

        print(best_sol)

        if len(best_sol) == 0:
            return {"solution": None, "value": 1}
        value = np.dot(self.c[:self.N*self.N], best_sol[:self.N*self.N])
        return {"solution": best_sol[:self.N*self.N], "value": value}

    def run_rec(self, best_sol, best_val_sol, curr_a, curr_b, curr_c):
        """

        """
        solver = Simplex(curr_a, curr_c, curr_b)
        solution = solver.run()
        #solution = linprog(curr_c, curr_a, curr_b, method="revised simplex")
        #solution = solution['x']

        if len(solution) == 0:
            return 1, []

        value_sol = np.dot(curr_c[:self.N*self.N], solution[:self.N*self.N])
        if value_sol <= best_val_sol and np.all([check_integer(x) for x in solution]) and np.all(np.array(solution) >= 0):
            return value_sol, copy.deepcopy(solution)

        if value_sol > best_val_sol:
            return 1, []

        current_best_sol = best_sol
        current_best_val = best_val_sol

        for i, value in enumerate(solution[:self.N*self.N]):
            if not check_integer(value):

                new_a, new_b, new_c = self.__add_constraint__(i, 0, curr_a, curr_b, curr_c)
                current_best_val, current_best_sol = self.run_rec(best_sol, best_val_sol, new_a, new_b, new_c)

                if current_best_val == value_sol and np.all([check_integer(x) for x in current_best_sol]) and np.all(np.array(current_best_sol) >= 0):
                    break

                new_a, new_b, new_c = self.__add_constraint__(i, 1, curr_a, curr_b, curr_c)
                val_sol_1, sol1 = self.run_rec(current_best_sol, current_best_val, new_a, new_b, new_c)

                if val_sol_1 < current_best_val:
                    current_best_val = val_sol_1
                    current_best_sol = sol1

                if current_best_val == 1:  # No possible solution
                    return 1, []

                break

        return current_best_val, current_best_sol

    def extract_path(self, solution):
        """
        :param solution: Result obtained from MissionPlanOptimizer.run, it should not be None.
        :return Optimum path, list.
        """
        assert (solution is not None), "Solution should not be None!"
        goal_matrix = np.reshape(solution, (self.N, self.N))
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
        self.c = list(self.c.flatten())

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

        combs = self.__create_combinations__()  # (8.1)
        for comb in combs:
            self.A.append(list(np.zeros(len(self.c))))
            self.b.append(len(comb)-1)
            for goal in comb:
                for j in range(len(self.R)):
                    if j in comb and j != goal:
                        self.A[-1][goal * len(self.R) + j] = 1

        for i in range(1, len(self.R)):  # (8.2)
            self.A.append(list(np.zeros(len(self.c))))
            self.b.append(1.)
            for j in range(len(self.R)):
                if j != i:
                    self.A[-1][i * len(self.R) + j] = -1.
            for j in range(len(self.R)):
                self.A[-1][i + j * len(self.R)] = 2.

        for i in range(len(self.A)):  # Adding slack variables
            for j in range(len(self.A)):
                if i == j:
                    self.A[i].append(1.)
                else:
                    self.A[i].append(0.)

        self.c = self.c + list(np.zeros(len(self.A[0]) - len(self.c)))
        self.c[self.N*self.N + 2] = 100

    def __create_combinations__(self) -> List:
        """
        Creates all possible combinations of goals 1..n-1
        @:return List of tuples, where each tuple is a combination of goals
        """
        combs = []
        goals = list(range(1, len(self.R)))
        for i in range(2, len(self.R)):
            combs += ([x for x in itertools.combinations(goals, i)])
        return combs

    def __add_constraint__(self, i, zero_or_one, curr_a, curr_b, curr_c):
        """
        Add integer constraints to variable i.
        :param i: index of variable with non integer value
        :param zero_or_one: This is a 0-1 integer programming problem, so the variables can be either 1s or 0s
        """
        new_b = copy.deepcopy(curr_b)
        new_a = np.array(curr_a)
        new_c = copy.deepcopy(curr_c)

        """
        if zero_or_one == 1:
            new_b[0] -= (self.T.flatten())[i]
            new_b[1] -= (self.E.flatten())[i]
            for j in range(2, len(new_b)):
                if new_a[j][i] != 0 and j != 4+i:
                    new_b[j] -= new_a[j][i]
                    if new_b[j] < 0:
                        new_b[j] = -new_b[j]
                        new_a[j] = -new_a[j]ì
        """
        new_c[self.N*self.N+4+i] = 100
        new_b[4+i] = zero_or_one

        return list(new_a), new_b, new_c


def check_integer(value):
    """
    Checks if given value is integer.
    :param value: float value
    :return True if integer, False otherwise
    """
    return value == int(value)
