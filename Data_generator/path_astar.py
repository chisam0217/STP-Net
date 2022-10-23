import numpy as np
import pyastar
import time
import os.path as op
import matplotlib.pyplot as plt
# The minimum cost must be 1 for the heuristic to be valid.
# The weights array must have np.float32 dtype to be compatible with the C++ code.
arr_dir = "../Data/2D_map/Map_array"
path_dir = "../Data/2D_map/Path"

def load_numpy():
	total_maz = np.load(op.join(arr_dir, 'maz.npy'))
	train_config = np.load(op.join(arr_dir, 'train_config.npy'))
	test_seen_config = np.load(op.join(arr_dir, 'test_seen_config.npy'))
	test_unseen_config = np.load(op.join(arr_dir, 'test_unseen_config.npy'))
	return total_maz, train_config, test_seen_config, test_unseen_config

def main():
	maz, train_config, test_seen_config, test_unseen_config = load_numpy()
	# maz.npy   20 x 100 x 100
	# train_config.npy  20 x 16000 x 4
	# test_seen_config.npy   20 x 4000 x 4
	# test_unseen_config.npy  4 x 2000 x 4
	s_time = time.time()
	train_path = []
	test_seen_path = []
	test_unseen_path = []

	train_numpath = []
	test_seen_numpath = []
	test_unseen_numpath = []
	direction = {(-1, 0): 1, (1, 0): 2, (0, -1):3, (0, 1): 4}

	max_y = 0
	min_y = 100

	for i in range(20):
		print ('Generating the path for maz', i)
		maz_i = maz[i]
		# for j in range(10): #train_config.shape[1]):
		for j in range(train_config.shape[1]):
			train_sg = train_config[i][j]
			# print ('train_sg', train_sg)
			path = pyastar.astar_path(maz_i, train_sg[:2], train_sg[2:], allow_diagonal=False)
			# print ('path', path)
			max_y = max(max_y, np.max(path))
			min_y = min(min_y, np.min(path))

			# for path_row in range(path.shape[0]):
			# 	if maz[i][path[path_row][0]][path[path_row][1]] == 255 or path[path_row][0]<0 or path[path_row][0] > 99 \
			# 		or path[path_row][1]<0 or path[path_row][1] > 99:
			# 		print ('111')


			# Draw some figure examples
			# draw_maz = np.copy(maz_i)
			# for pos in path:
			# 	draw_maz[pos[0], pos[1]] = 100
			# plt.imsave(op.join('vis', '{}.png'.format(j)), draw_maz)

			shift_path = np.zeros((path.shape[0], path.shape[1]), np.int32)
			shift_path[1:] = path[:-1]
			off_path = path - shift_path

			#record the path as numbers denoting directions
			num_path = []
			for row in range(1, off_path.shape[0]):
				num_path.append(direction[tuple(off_path[row])])

			# print ('num_path', num_path)
			train_numpath.append(num_path)

			# off_path = off_path.flatten()

			# print (off_path)
			# train_path.append(path[:-1].flatten().tolist())
			train_path.append(path.flatten().tolist())

		for j in range(test_seen_config.shape[1]):
			test_sg = test_seen_config[i][j]
			path = pyastar.astar_path(maz_i, test_sg[:2], test_sg[2:], allow_diagonal=False)
			# print ()

			shift_path = np.zeros((path.shape[0], path.shape[1]), np.int32)
			shift_path[1:] = path[:-1]
			off_path = path - shift_path

			#record the path as numbers denoting directions
			num_path = []
			for row in range(1, off_path.shape[0]):
				num_path.append(direction[tuple(off_path[row])])
	
			test_seen_numpath.append(num_path)

			# off_path = off_path.flatten()
			# print (off_path)
			# test_seen_path.append(path[:-1].flatten().tolist())
			test_seen_path.append(path.flatten().tolist())

	print ('max_y', max_y)
	print ('min_y', min_y)

	for i in range(20, 24):
		print ('Generating the path for maz', i)
		maz_i = maz[i]
		for j in range(test_unseen_config.shape[1]):
			test_sg = test_unseen_config[i - 20][j]
			path = pyastar.astar_path(maz_i, test_sg[:2], test_sg[2:], allow_diagonal=False)

			shift_path = np.zeros((path.shape[0], path.shape[1]), np.int32)
			shift_path[1:] = path[:-1]
			off_path = path - shift_path

			#record the path as numbers denoting directions
			num_path = []
			for row in range(1, off_path.shape[0]):
				num_path.append(direction[tuple(off_path[row])])
			
			test_unseen_numpath.append(num_path)

			# off_path = off_path.flatten()
			# print (off_path)
			# test_unseen_path.append(path[:-1].flatten().tolist())
			test_unseen_path.append(path.flatten().tolist())
			
	print ('train_path', len(train_path))
	print ('test_seen_path', len(test_seen_path))
	print ('test_unseen_path', len(test_unseen_path))
	print ('time', time.time() - s_time)

	train_file = op.join(path_dir, "train_path.txt")
	test_seen_file = op.join(path_dir, "test_seen_path.txt")
	test_unseen_file = op.join(path_dir, "test_unseen_path.txt")

	with open(train_file, 'w') as file:
		file.writelines('@'.join(str(j) for j in i) + '\n' for i in train_path)
	with open(test_seen_file, 'w') as file:
		file.writelines('@'.join(str(j) for j in i) + '\n' for i in test_seen_path)
	with open(test_unseen_file, 'w') as file:
		file.writelines('@'.join(str(j) for j in i) + '\n' for i in test_unseen_path)

	#record the path as numbers denoting directions
	train_num_file = op.join(path_dir, "train_numpath.txt")
	test_num_seen_file = op.join(path_dir, "test_seen_numpath.txt")
	test_num_unseen_file = op.join(path_dir, "test_unseen_numpath.txt")

	with open(train_num_file, 'w') as file:
		file.writelines('@'.join(str(j) for j in i) + '\n' for i in train_numpath)
	with open(test_num_seen_file, 'w') as file:
		file.writelines('@'.join(str(j) for j in i) + '\n' for i in test_seen_numpath)
	with open(test_num_unseen_file, 'w') as file:
		file.writelines('@'.join(str(j) for j in i) + '\n' for i in test_unseen_numpath)



if __name__ == '__main__':
	main()