"""
@Author: Carlo Cena

This module mimics the behavior of a client of the Mission_Optimizer (Integer Programming) package provided in this repository
Here is defined the method which receives the elaborated information on the known map and outputs the optimal sequence of goal to visit,
in order to maximise the prize and minimize the number of changes of direction.
"""
import time
import matplotlib.pyplot as plt
from Mission_Optimizer.mission_optimizer import MissionPlanOptimizer

show_animation = False  # Slower if true

# start and goal position
sx = -5  # [m], int
sy = -5  # [m], int
gx = [50, 10, 5, 50, 50, 55, 5, 0]  # [m], int
gy = [50, 30, 25, 55, 5, 30, -5, -2]  # [m], int
grid_size = 2  # [m], int
robot_radius = 1  # [m], int
vel = 2.0  # [m/s], float

Rewards = [0, 12, 10, 20, 30, 25, 40, 10, 20]  # int
Difficulties = [0, 0.5, 0, 0.2, 0.5, 1, 0.5, 0.3, 0.8]  # float between 0 and 1
max_t = 20  # [minutes], int
max_e = 90  # int

# set obstacle positions
ox, oy = [], []
for i in range(-10, 60):
    ox.append(i)
    oy.append(-10.0)
for i in range(-10, 60):
    ox.append(60.0)
    oy.append(i)
for i in range(-10, 61):
    ox.append(i)
    oy.append(60.0)
for i in range(-10, 61):
    ox.append(-10.0)
    oy.append(i)

# Square obstacle 1
for i in range(20, 40):
    ox.append(i)
    oy.append(20)
for i in range(20, 40):
    ox.append(i)
    oy.append(40)
for i in range(20, 40):
    ox.append(20)
    oy.append(i)
for i in range(20, 40):
    ox.append(40)
    oy.append(i)

# Square obstacle 2
for i in range(0, 20):
    ox.append(i)
    oy.append(0)
for i in range(0, 20):
    ox.append(i)
    oy.append(20)
for i in range(0, 20):
    ox.append(0)
    oy.append(i)
for i in range(0, 20):
    ox.append(20)
    oy.append(i)

if show_animation:  # pragma: no cover
    plt.plot(ox, oy, ".k")
    plt.plot(sx, sy, "og")
    plt.plot(gx, gy, "xr")
    plt.grid(True)
    plt.axis("equal")

optimizer1 = MissionPlanOptimizer(ox, oy, sx, sy, robot_radius, gx, gy, vel, Rewards, Difficulties, max_t, max_e, grid_size, show_animation=show_animation)
start_time = time.time()
result = optimizer1.run()
print("Total time, my version: ", time.time()-start_time)

if result["solution"] is None:
    print("No solution found.")
else:
    path = optimizer1.extract_path(result["solution"])
    print("Solution found: ", path)
    print("Value of solution: ", result["value"])
