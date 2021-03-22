# Mission Planning Optimization
Problem similar to prize-collecting TSP solved with implementation of branch &amp; bound and revised simplex algorithm
as a project for the course of Combinatorial Decision Making and Optimization (Master's degree in AI) of University of Bologna.

## Installation
Tested on python3.6 and python3.8.
```
python3 -m pip install numpy matplotlib
```

## Usage
The main class can be found in mission_optimizer.py, it requires a few information about the map and the agent (drone),
in particular, it needs:
- The coordinates of all the obstacles
- The coordinates of the starting position
- The coordinates of all the goals
- The reward and the difficulties associated to each goal
- The maximum time available to collect as much points as possible
- The maximum energy available (Optional for now, change the coefficient in the method "\_\_compute_energy_matrix\_\_")

All these parameters should be written in json files, as shown in "main.py".

In order to change the constraints one would need to work on the method "create_A_b", in "mission_optimizer.py".

The trajectories are computed by the script "a_star.py" taken from [PythonRobotics](https://github.com/AtsushiSakai/PythonRobotics)
and adapted in order to work with multiple goals, the script is called internally by the "MissionPlanOptimizer" class.
The path which maximizes the collected price, while minimizing the cost and keeping the energy 
and time requirements under the given thresholds, will be printed on the terminal.

## Details
More information can be found in the pdf report available in the repository.
