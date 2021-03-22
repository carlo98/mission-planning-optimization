"""
Utility methods for Mission Optimizer package.

Author: Carlo Cena
"""
import json


def check_integer(value):
    """
    Checks if given value is integer.
    :param value: float value
    :return True if integer, False otherwise
    """
    return value == int(value)


def build_map(json_path):
    """
    Builds a map from a json file. Expects the file to contain the vertices of each obstacle
    and of the walls, vertices should be given clockwise.
    :param json_path: path to the json file which describe the map
    :return ox, oy: lists of coordinates for obstacles and walls
    """
    ox, oy = [], []
    with open(json_path) as json_file:
        map_vertices = json.load(json_file)

    for elem_key in map_vertices:
        if elem_key == "grid_size":
            continue
        elem_coords = map_vertices[elem_key]
        for i in range(elem_coords["x"][0], elem_coords["x"][3]):  # Lower edge
            ox.append(i)
            oy.append(elem_coords["y"][0])
        for i in range(elem_coords["y"][3], elem_coords["y"][2]):  # Right-most edge
            ox.append(elem_coords["x"][3])
            oy.append(i)
        for i in range(elem_coords["x"][1], elem_coords["x"][2]):  # Upper edge
            ox.append(i)
            oy.append(elem_coords["y"][1])
        for i in range(elem_coords["y"][0], elem_coords["y"][1]):  # Left-most edge
            ox.append(elem_coords["x"][1])
            oy.append(i)

    return ox, oy, map_vertices["grid_size"]
