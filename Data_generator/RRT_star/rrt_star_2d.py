# This file is subject to the terms and conditions defined in
# file 'LICENSE', which is part of this source code package.
import numpy as np
import sys
import os
# print (os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
#                 "/../../src/")
# sys.path.append("../../")
# from src.rrt.rrt_star import RRTStar
from rrt_star import RRTStar
from search_space import SearchSpace
from utilities.plotting import Plot




# sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                # "/../../Sampling_based_Planning/")


X_dimensions = np.array([(0, 128), (0, 128)])  # dimensions of Search Space
# obstacles
Obstacles = np.array([(20, 20, 40, 40), (20, 60, 40, 80), (60, 20, 80, 40), (60, 60, 80, 80)])
x_init = (0, 0)  # starting location
x_goal = (100, 100)  # goal location

# Q = np.array([(8, 4)])  # length of tree edges
Q = np.array([(8,4)])
r = 1  # length of smallest edge to check for intersection with obstacles
max_samples = 1024  # max number of samples to take before timing out
rewire_count = 32  # optional, number of nearby branches to rewire
prc = 0.1  # probability of checking for a connection to goal

# create Search Space
X = SearchSpace(X_dimensions, Obstacles)

# create rrt_search
rrt = RRTStar(X, Q, x_init, x_goal, max_samples, r, prc, rewire_count)
path = rrt.rrt_star()
path = np.array(path, dtype=np.int)

# path = int(path)
print ('path', path)

# plot
plot = Plot("rrt_star_2d")
plot.plot_tree(X, rrt.trees)
if path is not None:
    plot.plot_path(X, path)
plot.plot_obstacles(X, Obstacles)
plot.plot_start(X, x_init)
plot.plot_goal(X, x_goal)
# print (path)
plot.draw()
# plot.draw(auto_open=True)
