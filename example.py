"""
@Author: Carlo Cena

This module mimics the behavior of a client of the Mission_Optimizer (Integer Programming) package provided in this repository
Here is defined the method which receives the elaborated information on the known map and outputs the optimal sequence of goal to visit,
in order to maximise the prize and minimize the number of changes of direction.
"""
from Mission_Optimizer.mission_optimizer import MissionPlanOptimizer

Time = [[0, 5, 10, 10, 7],
        [5, 0, 10, 10, 10],
        [10, 10, 0, 5, 5],
        [10, 10, 5, 0, 5],
        [7, 10, 5, 5, 0]]
Energy = [[0, 1, 1, 1, 1],
          [1, 0, 1, 1, 1],
          [1, 1, 0, 1, 1],
          [1, 1, 1, 0, 1],
          [1, 1, 1, 1, 0]]
Changes = [[0, 5, 10, 10, 7],
           [5, 0, 10, 10, 10],
           [10, 1, 0, 2, 5],  # Play with 2 element of this row to see changes in result and test cycles
           [10, 10, 5, 0, 2],
           [7, 10, 2, 5, 0]]
Rewards = [0, 6, 6, 6, 6]
Difficulties = [0, 0.5, 0.5, 0.5, 0.5]
max_t = 20
max_e = 90

optim = MissionPlanOptimizer(Time, Energy, Changes, Rewards, Difficulties, max_t, max_e)
print("C matrix")
for i in range(len(optim.C)):
    print(optim.C[i])

result = optim.run()

# Accessing variables
if result is not None:
    print("Edges")
    for i in range(len(result["G"])):
        print(result["G"][i])
    print("Total time")
    print(result["tot_time"])
    print("Total energy")
    print(result["tot_energy"])
    print("Minimized value")
    print(result["value"])
    print("Optimum path")
    optim_path = optim.extract_path(result)
    print(optim_path)
    r = 0
    for i in optim_path:
        r += Rewards[i]
    print("Collected Reward")
    print(r)
else:
    print("No solution has been found.")
