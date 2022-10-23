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

arr_dir = "../../Data/3D_map/Map_array"
# total_maz = np.load(op.join(arr_dir, 'maz.npy'))
obs = np.load(op.join(arr_dir, 'obs.npy'))
# train_config = np.load(op.join(arr_dir, 'train_config.npy'))
test_seen_config = np.load(op.join(arr_dir, 'test_seen_config.npy'))
# test_unseen_config = np.load(op.join(arr_dir, 'test_unseen_config.npy'))

# test_seen_path = np.load(op.join(arr_dir, 'test_seen_path.npy'))
# sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                # "/../../Sampling_based_Planning/")

# test_seen_path = np.reshape(test_seen_path, (100, -1, test_seen_path.shape[1], test_seen_path.shape[2]))
# print ('test_seen_path', test_seen_path.shape)
i,j = 0,1
Obstacles = obs[i]

X_dimensions = np.array([(0, 19), (0, 19), (0, 19)])

x_init = test_seen_config[i,j,:3]
x_goal = test_seen_config[i,j,3:]


X = SearchSpace(X_dimensions, Obstacles)
path = [(7, 10, 15), (9, 8, 14), (18, 1, 14), (18, 0, 15)]
# path = test_seen_path[i,j]


plot = Plot("rrt_star_3d")
# plot.plot_tree(X, rrt.trees)
if path is not None:
    plot.plot_path(X, path)
    plot.plot_obstacles(X, Obstacles)
    plot.plot_start(X, x_init)
    plot.plot_goal(X, x_goal)
    # print (path)
    plot.draw()

