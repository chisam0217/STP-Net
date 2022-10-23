# This file is subject to the terms and conditions defined in
# file 'LICENSE', which is part of this source code package.
import numpy as np

# from src.rrt.rrt_star import RRTStar
# from src.search_space.search_space import SearchSpace
# from src.utilities.plotting import Plot
from rrt_star import RRTStar
from search_space import SearchSpace
from utilities.plotting import Plot
import os.path as op
import time

arr_dir = "../../Data/3D_map/Map_array"

obs = np.load(op.join(arr_dir, 'obs.npy'))
train_config = np.load(op.join(arr_dir, 'train_config.npy'))

def RRTstar(Obstacles, configuration):
    X_dimensions = np.array([(0, 19), (0, 19), (0, 19)])  # dimensions of Search Space
    x_init = (configuration[0], configuration[1], configuration[2])

    x_goal = (configuration[3], configuration[4], configuration[5])

    print ('x_init', x_init)
    print ('x_goal', x_goal)


    Q = np.array([(2,1)])  # length of tree edges
    r = 1  # length of smallest edge to check for intersection with obstacles
    max_samples = 1024  # max number of samples to take before timing out
    rewire_count = 3  # optional, number of nearby branches to rewire
    prc = 0.1  # probability of checking for a connection to goal

    # create Search Space
    X = SearchSpace(X_dimensions, Obstacles)

# create rrt_search
    rrt = RRTStar(X, Q, x_init, x_goal, max_samples, r, prc, rewire_count)
    path = rrt.rrt_star()

    path = np.array(path, dtype=np.int)

    print ('path', path)

# plot
    plot = Plot("rrt_star_3d")
    plot.plot_tree(X, rrt.trees)
    if path is not None:
        plot.plot_path(X, path)
    plot.plot_obstacles(X, Obstacles)
    plot.plot_start(X, x_init)
    plot.plot_goal(X, x_goal)
    plot.draw(auto_open=True)
    return path



s_time = time.time()
map100_train = []
# for i in range(100):
#     print ('map', i)
#     for j in range(5):
#     # for j in range(train_config.shape[1]):
#         # print ('j', j)
#         # print ('train_config[i,j]', train_config[i,j])
#         waypoints = RRTstar(obs[i], train_config[i,j])
# #         print ('waypoints', waypoints.shape[0])
#         map100_train.append(waypoints)

i, j = 0,100
waypoints = RRTstar(obs[i], train_config[i,j])
#         print ('waypoints', waypoints.shape[0])
map100_train.append(waypoints)

print ('The total time is', time.time() - s_time)


# In[10]:


from statistics import median
train_paths = map100_train.copy()
train_seq_len = [path.shape[0] for path in train_paths]
max_train_seq_len = max(train_seq_len)
for i in range(len(train_paths)):
    curr_seq_len = train_paths[i].shape[0]
    pad_seq_len = max_train_seq_len - curr_seq_len
    pad_seq = np.tile(train_paths[i][-2], (pad_seq_len, 1))
    train_paths[i] = np.concatenate((train_paths[i], pad_seq),axis = 0)
train_paths = np.array(train_paths)
print (train_paths.shape)
    
train_file = op.join(arr_dir, "train_path.npy")
np.save(train_file, train_paths)







map100_test_seen = []
for i in range(100):
    print ('map', i)
    for j in range(test_seen_config.shape[1]):
        # print ('j', j)
        # print ('train_config[i,j]', train_config[i,j])
        waypoints = RRTstar(obs[i], test_seen_config[i,j])
        map100_test_seen.append(waypoints)


# In[12]:


test_seen_paths = map100_test_seen.copy()
test_seen_seq_len = [path.shape[0] for path in test_seen_paths]
max_test_seen_seq_len = max(test_seen_seq_len)
for i in range(len(test_seen_paths)):
    curr_seq_len = test_seen_paths[i].shape[0]
    pad_seq_len = max_test_seen_seq_len - curr_seq_len
    pad_seq = np.tile(test_seen_paths[i][-2], (pad_seq_len, 1))
    test_seen_paths[i] = np.concatenate((test_seen_paths[i], pad_seq),axis = 0)
test_seen_paths = np.array(test_seen_paths)
print (test_seen_paths.shape)
    
test_seen_file = op.join(arr_dir, "test_seen_path.npy")
np.save(test_seen_file, test_seen_paths)


# In[13]:


map20_test_unseen = []
for i in range(100, 120):
    print ('map', i)
    for j in range(test_unseen_config.shape[1]):
        # print ('j', j)
        # print ('train_config[i,j]', train_config[i,j])
        waypoints = RRTstar(obs[i], test_unseen_config[i-100,j])
        map20_test_unseen.append(waypoints)


# In[14]:


test_unseen_paths = map20_test_unseen.copy()
test_unseen_seq_len = [path.shape[0] for path in test_unseen_paths]
max_test_unseen_seq_len = max(test_unseen_seq_len)
for i in range(len(test_unseen_paths)):
    curr_seq_len = test_unseen_paths[i].shape[0]
    pad_seq_len = max_test_unseen_seq_len - curr_seq_len
    pad_seq = np.tile(test_unseen_paths[i][-2], (pad_seq_len, 1))
    test_unseen_paths[i] = np.concatenate((test_unseen_paths[i], pad_seq),axis = 0)
test_unseen_paths = np.array(test_unseen_paths)
print (test_unseen_paths.shape)
    
test_unseen_file = op.join(arr_dir, "test_unseen_path.npy")
np.save(test_unseen_file, test_unseen_paths)
