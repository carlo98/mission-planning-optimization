"""
@Author: Carlo Cena

This module mimics the behavior of a client of the Mission_Optimizer (Integer Programming) package provided in this repository
Here is defined the method which receives the elaborated information on the known map and outputs the optimal sequence of goal to visit,
in order to maximise the prize and minimize the number of changes of direction.
"""
from Mission_Optimizer.mission_optimizer import MissionPlanOptimizer

Time = [[0, 1, 1],
        [1, 0, 1],
        [1, 1, 0]]
Energy = [[0, 1, 1],
          [1, 0, 1],
          [1, 1, 0]]
Changes = [[0, 1, 1],
           [1, 0, 6],
           [5, 4, 0]]
Rewards = [0, 6, 5]
Difficulties = [0, 0, 0]
max_t = 20
max_e = 90

optim = MissionPlanOptimizer(Time, Energy, Changes, Rewards, Difficulties, max_t, max_e)

result = optim.run()
