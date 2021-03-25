# Mission Planning Optimization
Problem similar to prize-collecting TSP solved with implementation of branch &amp; bound and revised simplex algorithm
as a project for the course of Combinatorial Decision Making and Optimization (Master's degree in AI) of University of Bologna.

# Table of contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Example](#example)
4. [JSON files](#json)
    1. [Map and Grid Size](#map_grid)
    2. [Problem Parameters](#problem_par)
    3. [Agent Parameters](#agent_par)
5. [Details](#details)    
    
## Installation <a name="installation"></a>
Tested on python3.6 and python3.8.
```
python3 -m pip install numpy matplotlib
```

## Usage <a name="usage"></a>
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

## Example <a name="example"></a>
An example of usage is given in the script main.py:
```
python3 main.py
```

## JSON files <a name="json"></a>
In the following paragraph are describe the json files used to represent map, obstacles, goals and drone parameters.
### Map and Grid Size <a name="map_grid"></a>
The map description, along with the grid size with which to discretize the map, should be written in "map.json", as:
- A dictionary, named "walls", which contains two lists of integers of coordinates for the vertices of the walls.
```"walls": {"x": [0, 0, 100, 100],"y": [0, 100, 100, 0]}```
- A dictionary, named "obs_n", which contains two lists of integers of coordinates for the vertices of the n-th obstacle, for each obstacle.
```"obs_1": {"x": [10, 10, 30, 30],"y": [10, 30, 30, 10]}```
- An integer parameter, called "grid_size", which is used as value for the discretization of the map.

### Problem Parameters <a name="problem_par"></a>
The problem parameters, i.e. goals' coordinates, rewards, difficulties and max time and energy, should be written 
in the file "problem_par.json", as:

- A list of integers, named "goals_x", with the x-coordinates of the goals
- A list of integers, named "goals_y", with the y-coordinates of the goals
- A list of integers, named "rewards", with the reward associated to each goal
- A list of float between 0 and 1, named "difficulties", with the difficulties associated to each goal
- An integer, called "max_t", which represents the maximum time available
- An integer, called "max_e", which represents the maximum energy available

### Agent Parameters <a name="agent_par"></a>
The file "agent_par.json" contains the parameters associated to the agent, such as:
- An integer, called "robot_radius", which represents the radius of the agent
- A float, called "vel", that represents the constant velocity of the agent

## Details <a name="details"></a>
More information can be found in the pdf report available in the repository.
