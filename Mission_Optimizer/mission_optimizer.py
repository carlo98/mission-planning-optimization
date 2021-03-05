"""
Optimizes plan of an agent by choosing the optimum order of goal to visit in order to maximize the reward obtained while
minimizing the costs required and staying below the maximum time and energy available.

Author: Carlo Cena
"""
from typing import List


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
        if not isinstance(time_matrix, List) or not isinstance(energy_matrix, List) or not isinstance(cost_matrix, List)\
                or not isinstance(rewards, List) or not isinstance(difficulties, List) or not isinstance(max_t, int)\
                or not isinstance(max_e, int):
            print("Check inputs' type.")
            raise TypeError
        self.T = time_matrix
        self.E = energy_matrix
        self.C = cost_matrix
        self.max_t = max_t
        self.max_e = max_e
        self.N = len(self.E)
        r = rewards
        d = difficulties
        self.__compute_cost_matrix__(r, d)

    def run(self):
        """
        Uses branch and bound and simplex class to get optimal solution of
        constrained optimization problem.
        :return result.solution is the best solution, if no solution exists result.solution is None.
        """
        pass
        
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

    def __compute_cost_matrix__(self, reward, difficulty):
        """
        :param reward: reward array
        :param difficulty: difficulty array
        """
        for i in range(self.N):
            for j in range(self.N):
                if i != j:  # We don't want loops
                    self.C[i][j] -= reward[i] * difficulty[i]
