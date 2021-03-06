"""
Example of use of Mission Optimizer package.
"""
import matplotlib.pyplot as plt
import json
import time
from Mission_Optimizer.mission_optimizer import MissionPlanOptimizer
from Mission_Optimizer.utility import build_map
import logging

logging.basicConfig(level=logging.INFO)
show_animation = False  # Slower If true

# Starting position
sx = 5  # [m], int
sy = 5  # [m], int

with open("problem_par.json") as json_file:
    problem_par = json.load(json_file)
    # start and goal position
    gx = problem_par["goals_x"]  # [m], int
    gy = problem_par["goals_y"]  # [m], int
    Rewards = problem_par["rewards"]  # int
    Difficulties = problem_par["difficulties"]  # float between 0 and 1
    max_t = problem_par["max_t"]  # [minutes], int
    max_e = problem_par["max_e"]  # int

with open("agent_par.json") as json_file:
    agent_par = json.load(json_file)
    robot_radius = agent_par["robot_radius"]  # [m], int
    vel = agent_par["vel"]  # [m/s], float
    costs_changes = agent_par["costs_changes"]

# Set obstacle positions
ox, oy, grid_size = build_map("map.json")

if show_animation:  # pragma: no cover
    plt.plot(ox, oy, ".k")
    plt.plot(sx, sy, "og")
    plt.plot(gx, gy, "xr")
    plt.grid(True)
    plt.axis("equal")

if max(Rewards) == 0 or min(Rewards) < 0:
    print("Check rewards, all values should be >= 0 and at least one should be > 0.")
    exit(-1)

if max(Difficulties) > 1 or min(Difficulties) < 0 or max(Difficulties) == 0:
    print("Check difficulties, range should be 0-1 and at least one value should be greater then 0.")
    exit(-1)

optimizer1 = MissionPlanOptimizer(ox, oy, sx, sy, robot_radius, gx, gy, vel, Rewards, Difficulties, max_t, max_e, grid_size, costs_changes, show_animation=show_animation)

start_time = time.time()
result = optimizer1.run()
end_time = time.time()

# Accessing variables
if result["solution"] is not None:
    optimizer1_path = optimizer1.extract_path(result["solution"])

    r = 0
    for i in optimizer1_path:
        r += Rewards[i]
    print("Collected Reward: ", r)

    print("Minimized value")
    print(str(result["value"]))
    print("Optimum path")
    print(optimizer1_path)
else:
    print("No bounded solution has been found.")

print("Time for solution: " + str(end_time-start_time))
