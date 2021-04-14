"""
Optimizes plan of an agent by choosing the optimum order of goal to visit in order to maximize the reward obtained while
minimizing the costs required and staying below the maximum time and energy available.

Author: Carlo Cena
"""
from typing import List
import itertools
import copy
from heapq import heappop, heappush
import numpy as np
import logging
import time

from Mission_Optimizer.a_star import AStarPlanner
from Mission_Optimizer.simplex import Simplex
from Mission_Optimizer.cost_functions import compute_cost_vector, compute_energy_matrix
from Mission_Optimizer.utility import check_integer


class MissionPlanOptimizer:

    def __init__(self, ox: List[int], oy: List[int], sx: int, sy: int, rr: int, gx: List[int], gy: List[int],
                 vel: float, rewards: List[int], difficulties: List[float], max_t: int, max_e: int, resolution: int, costs_changes: dict,
                 show_animation=False):
        """
        @:param oy: list of obstacles' y coordinate.
        @:param sx: x coordinate of start position.
        @:param sy: y coordinate of start position.
        @:param rr: radius of agent.
        @:param gx: list of goals' x coordinate.
        @:param gy: list of goals' y coordinate.
        @:param vel: velocity of agent.
        @:param rewards: list (n+1) where R[i] is the reward obtained by visiting goal i.
        @:param difficulties: list (n+1) where D[i] is the difficulty associated to goal i.
        @:param max_t: Integer, maximum time available in seconds.
        @:param max_e: Integer 0-100, maximum energy available.
        @:param resolution: Resolution of grid.
        @:param costs_changes: Dictionary with cost associated to each number of changes of direction.
        @:param show_animation: Whether to show or not the animation for trajectory planning.
        With n = number of goals + 1 (starting position).
        """
        if not isinstance(rewards, List) or not isinstance(difficulties, List) or not isinstance(max_t, int) or not isinstance(max_e, int)\
                or not isinstance(ox, List) or not isinstance(oy, List) or not isinstance(sx, int) or not isinstance(sy, int) or not isinstance(rr, int)\
                or not isinstance(gx, List) or not isinstance(gy, List) or not isinstance(vel, float) or not isinstance(costs_changes, dict):
            logging.error("Check inputs' type.")
            raise TypeError
        trajectory_planner = AStarPlanner(ox, oy, resolution, rr, show_animation=show_animation)
        self.T, self.Costs = trajectory_planner.planning(sx, sy, gx, gy, vel)
        self.E = compute_energy_matrix(self.T)
        self.max_t = max_t
        self.max_e = max_e
        self.R = [0.] + rewards
        self.D = [0.] + difficulties
        self.N = len(self.R)
        self.costs_changes = costs_changes
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
        counter = itertools.count()
        best_sol = []

        node = (self.b, self.c)
        problems = [(next(counter), node)]  # Initial problem
        best_val_sol = 0
        n_squared = self.N*self.N

        while len(problems) > 0:
            num_prob, node = heappop(problems)
            logging.info("Problem number: %d", num_prob)
            solver = Simplex(self.A, node[0], node[1])
            start_time = time.time()
            solution = solver.run()  # Solving current problem
            logging.info("\tTime fot intermediate solution: %.2f", time.time()-start_time)
            if len(solution) == 0:  # No bounded solution
                logging.info("\tProblem has no bounded solution")
                continue
            value_sol = np.dot(self.c[:n_squared], solution[:n_squared])
            value_sol = np.around(value_sol, decimals=3)  # Avoiding small gains
            logging.info("\tResult: %.3f Best result: %.3f", value_sol, best_val_sol)

            if value_sol >= best_val_sol:  # Solution worst then best integer one found so far
                logging.info("\tSolution worst or equal then best integer one found so far")
                continue
            elif np.all([check_integer(x) for x in solution[:n_squared]]):  # Integer solution
                logging.info("\tBest integer solution found so far")
                best_sol = solution[:n_squared]
                best_val_sol = value_sol
            else:
                for i, value in enumerate(solution[:n_squared]):
                    if not check_integer(value):
                        logging.info("\tBranching on variable %d", i)
                        new_b, new_c = self.__add_constraint__(i, 0, node[0], node[1])
                        new_node = (new_b, new_c)
                        heappush(problems, (next(counter), new_node))
                        new_b, new_c = self.__add_constraint__(i, 1, node[0], node[1])
                        new_node = (new_b, new_c)
                        heappush(problems, (next(counter), new_node))
                        break

        if len(best_sol) == 0:  # No bounded solution found
            logging.warning("No bounded solution has been found.")
            return {"solution": None, "value": 1}
        value = np.dot(self.c[:n_squared], best_sol)
        return {"solution": best_sol, "value": value}

    def extract_path(self, solution):
        """
        :param solution: Result obtained from MissionPlanOptimizer.run, it should not be None.
        :return Optimum path, list.
        """
        assert (solution is not None), "Solution should not be None!"
        goal_matrix = np.reshape(solution, (self.N, self.N))
        optimum_path = []
        for i in range(1, self.N):  # First edge
            if goal_matrix[i][0] == 1:
                optimum_path.append(i)
                break
        if len(optimum_path) == 0:  # No path from starting position
            logging.warning("No path from starting position found.")
            return []

        flag = True
        while flag:
            flag = False
            for i in range(1, self.N):
                if goal_matrix[i][optimum_path[-1]] == 1:
                    optimum_path.append(i)
                    flag = True
                    break

        optimum_path = [x-1 for x in optimum_path]  # Removing index of starting position
        return optimum_path

    def __simplex_structures__(self):
        """
        Creates matrix A and vectors b and c for simplex algorithm.
        """
        self.c = compute_cost_vector(self.Costs, self.N, self.R, self.D, self.costs_changes)
        self.__create_A_b__()
        logging.info("Number of constraints: %d", len(self.A))

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

        self.A.append(list(np.zeros(len(self.c))))  # (3)
        self.b.append(1.)
        for i in range(self.N):
            self.A[-1][i * self.N] = 1.

        self.A.append(list(np.zeros(len(self.c))))  # (4)
        self.b.append(0.)
        for i in range(self.N):
            self.A[-1][i] = 1.

        for i in range(len(self.c)):  # (5)
            self.A.append(list(np.zeros(len(self.c))))
            self.A[-1][i] = 1.
            self.b.append(1.)

        for i in range(1, self.N):  # (6)
            self.A.append(list(np.zeros(len(self.c))))
            self.b.append(1.)
            for j in range(self.N):
                self.A[-1][i * self.N + j] = 1.

        for i in range(self.N):  # (7)
            self.A.append(list(np.zeros(len(self.c))))
            self.b.append(1.)
            for j in range(self.N):
                self.A[-1][i + j * self.N] = 1.

        combs = self.__create_combinations__()  # (8.1)
        for comb in combs:
            self.A.append(list(np.zeros(len(self.c))))
            self.b.append(len(comb)-1)
            for goal in comb:
                for j in range(self.N):
                    if j in comb and j != goal:
                        self.A[-1][goal * self.N + j] = 1

        for i in range(1, self.N):  # (8.2)
            self.A.append(list(np.zeros(len(self.c))))
            self.b.append(1.)
            for j in range(self.N):
                if j != i:
                    self.A[-1][i * self.N + j] = -1.
            for j in range(self.N):
                self.A[-1][i + j * self.N] = 2.

        for i in range(len(self.A)):  # Adding slack variables
            for j in range(len(self.A)):
                if i == j:
                    self.A[i].append(1.)
                else:
                    self.A[i].append(0.)

        tmp_len = len(self.c)
        self.c = self.c + list(np.zeros(len(self.A[0]) - len(self.c)))

        self.c[tmp_len + 2] = 10000  # 3, Big M method

    def __create_combinations__(self) -> List:
        """
        Creates all possible combinations of goals 1..n-1
        @:return List of tuples, where each tuple is a combination of goals
        """
        combs = []
        goals = list(range(1, self.N))
        for i in range(2, self.N):
            combs += ([x for x in itertools.combinations(goals, i)])
        return combs

    def __add_constraint__(self, i, zero_or_one, curr_b, curr_c):
        """
        Add integer constraints to variable i.
        :param i: index of variable with non integer value
        :param zero_or_one: This is a 0-1 integer programming problem, so the variables can be either 1s or 0s
        """
        new_b = copy.deepcopy(curr_b)
        new_c = copy.deepcopy(curr_c)

        new_c[self.N*self.N+4+i] = 10000  # Big M method
        new_b[4+i] = zero_or_one

        return new_b, new_c
