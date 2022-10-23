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
import os.path as op
import time

arr_dir = "../../Data/2D_map/Map_array"
total_maz = np.load(op.join(arr_dir, 'maz.npy'))
obs = np.load(op.join(arr_dir, 'obs.npy'))
train_config = np.load(op.join(arr_dir, 'train_config.npy'))
test_seen_config = np.load(op.join(arr_dir, 'test_seen_config.npy'))
test_unseen_config = np.load(op.join(arr_dir, 'test_unseen_config.npy'))


# sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                # "/../../Sampling_based_Planning/")

def RRTstar(Obstacles, configuration):

    X_dimensions = np.array([(0, 128), (0, 128)])  # dimensions of Search Space
# obstacles
# Obstacles = np.array([(20, 20, 40, 40), (20, 60, 40, 80), (60, 20, 80, 40), (60, 60, 80, 80)])
# print ('Obstacles', Obstacles.shape)
# x_init = (0, 0)  # starting location
# x_goal = (100, 100)  # goal location

# print ('obs', obs.shape)
# Obstacles = obs[6]
# print ('Obstacles', Obstacles.shape)
# x_init = (train_config[0,10,0], train_config[0,10,1])  # starting location
# x_goal = (train_config[0,10,2], train_config[0,10,3])  # goal location
    x_init = (configuration[0], configuration[1])
    x_goal = (configuration[2], configuration[3])
# x_init = (0, 0)  # starting location
# x_goal = (126, 126)  # goal location


    Q = np.array([(8, 4)])  # length of tree edges
    # Q = np.array([(4,2)])
    # Q = np.array([(6,3)])
    r = 1  # length of smallest edge to check for intersection with obstacles
    max_samples = 100000  # max number of samples to take before timing out
    rewire_count = 8#4  # optional, number of nearby branches to rewire
    prc = 0.1  # probability of checking for a connection to goal
    # prc = 0.5

# create Search Space
    X = SearchSpace(X_dimensions, Obstacles)

# create rrt_search
    rrt = RRTStar(X, Q, x_init, x_goal, max_samples, r, prc, rewire_count)
    path = rrt.rrt_star()
    path = np.array(path, dtype=np.int)


    print ('path', path)

    # plot = Plot("rrt_star_2d")
    # plot.plot_tree(X, rrt.trees)
    # if path is not None:
    #     plot.plot_path(X, path)
    #     plot.plot_obstacles(X, Obstacles)
    #     plot.plot_start(X, x_init)
    #     plot.plot_goal(X, x_goal)
    #     # print (path)
    #     plot.draw()


    return path

s_time = time.time()
# print ('total_maz', total_maz.shape)
# print ('train_config[0, 15]', train_config[0, 15])
# print ('111', total_maz[0,67,21])
# waypoints = RRTstar(obs[0], train_config[0, 2])

# print ('train_config', train_config.shape)

all_waypoints = []
for i in range(1):
    # for j in range(train_config.shape[1]):
    for j in range(500):
        print ('j', j)
        # print ('train_config[i,j]', train_config[i,j])
        waypoints = RRTstar(obs[i], train_config[i,j])
        all_waypoints.append(waypoints)
all_waypoints = np.array(all_waypoints)
# print ('all_waypoints', all_waypoints.shape)
# print ('the Max', np.max(all_waypoints))
# print ('the Min', np.min(all_waypoints))

print ('The total time is', time.time() - s_time)


# # plot
# plot = Plot("rrt_star_2d")
# plot.plot_tree(X, rrt.trees)
# if path is not None:
#     plot.plot_path(X, path)
# plot.plot_obstacles(X, Obstacles)
# plot.plot_start(X, x_init)
# plot.plot_goal(X, x_goal)
# # print (path)
# plot.draw()
# # plot.draw(auto_open=True)
