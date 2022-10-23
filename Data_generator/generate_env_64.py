import numpy as np 
import random
import matplotlib.pyplot as plt
import os.path as op

pic_dir = "../Data/2D_map_64_new/Map_pictures"
arr_dir = "../Data/2D_map_64_new/Map_array"
num_workspace = 120

# maz.npy   (120, 128, 128)
# train_config.npy  (100, 1600, 4)
# test_seen_config.npy   (100, 400, 4)
# test_unseen_config.npy  (20, 400, 4)
# obs.npy (120, 10, 4)


# maz.npy   20 x 100 x 100
# train_config.npy  20 x 16000 x 4
# test_seen_config.npy   20 x 4000 x 4
# test_unseen_config.npy  4 x 2000 x 4



np.random.seed(23)
# #select the centers
# def generate_2D_map(size=100, num_config=16000, maz_id=0):
# 	maz = np.ones((size, size)) 

# 	# obs_style = [[5,5], [7,7], [5,7], [7,5], [9,9], [9, 7], [7, 9], [11,5]]
# 	obs_style = [[5,5], [7,7], [9,9], [11,11], [13,13]]
# 	# obs_style = [[10,10], [15,15],[20,20],[25,25]]
# 	# obs_style = [[10,10], [14,14], [10,14], [14,10], [18,18], [18, 14], [14, 18], [22,10]]

# 	# obs_num = random.randint(8, 12)
# 	obs_num = 12#15
# 	obs_pos = np.random.randint(low=7, high=56, size=(obs_num, 2)) 
# 	#randomize the central positions of obstacles, avoid out of range of the array
# 	obs_bound = np.zeros((obs_num, 4)) 

# 	for i in range(obs_pos.shape[0]):
# 		a, b = obs_pos[i]
# 		random_style = random.randint(0, len(obs_style)-1)
# 		h, w = obs_style[random_style]
# 		h, w = h//2, w//2
# 		maz[a-h:a+h, b-w:b+w] = 255
# 		obs_bound[i][0] = a-h
# 		obs_bound[i][1] = b-w
# 		obs_bound[i][2] = a+h
# 		obs_bound[i][3] = b+w

# 	def generate_sg(nc, space):
# 		s_g_mat = []

# 		for i in range(nc):
# 			# s_pos = np.random.randint(low=0, high=127, size=(2)) 
# 			# g_pos = np.random.randint(low=0, high=127, size=(2)) 
			

# 			while True:
# 				s_pos = np.random.randint(low=0, high=63, size=(2)) 
# 				g_pos = np.random.randint(low=0, high=63, size=(2)) 
# 				if space[s_pos[0], s_pos[1]] == 255 or space[g_pos[0], g_pos[1]] == 255 or np.sum(np.absolute(s_pos - g_pos)) < 100: 
# 					continue
# 				flag = True
# 				four_dir = np.array([[-1,-1],[-1,1],[1,-1],[1,1]])
# 				for direction in four_dir:
# 					tmp_s_pos = direction + s_pos
# 					tmp_g_pos = direction + g_pos
# 					if all(tmp_s_pos >=0) and space[tmp_s_pos[0], tmp_s_pos[1]] == 255:
# 						flag = False
# 					if all(tmp_g_pos >=0) and space[tmp_g_pos[0], tmp_g_pos[1]] == 255:
# 						flag = False
# 					# if all(tmp_s_pos >= 0) and all(tmp_g_pos >=0):
# 					# 	if space[tmp_s_pos[0], tmp_s_pos[1]] == 255 or space[tmp_g_pos[0], tmp_g_pos[1]] == 255:
# 					# 		Flag = False
# 				if flag:
# 					break



# 			# while space[s_pos[0], s_pos[1]] == 255 or space[g_pos[0], g_pos[1]] == 255 or np.sum(np.absolute(s_pos - g_pos)) < 30: 
# 			# 	four_dir = np.array([[-1,-1],[-1,1],[1,-1],[1,1]])
# 			# 	for direction in four_dir:
# 			# 		tmp_s_pos = direction + s_pos
# 			# 		tmp_g_pos = direction + g_pos
# 			# 		if all(tmp_s_pos >= 0) and all(tmp_g_pos >=0):
# 			# 			if space[tmp_s_pos[0], tmp_s_pos[1]] == 255 or space[tmp_g_pos[0], tmp_g_pos[1]] == 255:
# 			# 				Flag = False

# 			# 	# space[s_pos[0]-1, s_pos[1]-1] == 255 or space[s_pos[0]-1, s_pos[1]+1] == 255 or space[s_pos[0]+1, s_pos[1]-1] == 255 or space[s_pos[0]+1, s_pos[1]+1] == 255 \
# 			# 	# space[g_pos[0]-1, g_pos[1]-1] == 255 or g_pos[g_pos[0]-1, g_pos[1]+1] == 255 or space[g_pos[0]+1, g_pos[1]-1] == 255 or space[g_pos[0]+1, g_pos[1]+1] == 255:
# 			# 	s_pos = np.random.randint(low=0, high=127, size=(2)) 
# 			# 	g_pos = np.random.randint(low=0, high=127, size=(2)) 
# 			# # space[s_pos[0], s_pos[1]] = 2
# 			# space[g_pos[0], g_pos[1]] = 3
# 			s_g_mat.append(np.concatenate((s_pos, g_pos), axis=None))

# 		return np.stack(s_g_mat)

# 	save_maz(maz, maz_id)
# 	s_g_maz = generate_sg(num_config, maz)

# 	return maz, s_g_maz, obs_bound


def generate_2D_map(size=100, num_config=16000, maz_id=0):
	maz = np.ones((size, size)) 

	# obs_style = [[5,5], [7,7], [5,7], [7,5], [9,9], [9, 7], [7, 9], [11,5]]
	obs_style = [[7,7], [9,9], [11,11]]
	# obs_style = [[10,10], [15,15],[20,20],[25,25]]
	# obs_style = [[10,10], [14,14], [10,14], [14,10], [18,18], [18, 14], [14, 18], [22,10]]

	# obs_num = random.randint(8, 12)
	obs_num = 8#15
	obs_pos = np.random.randint(low=0, high=50, size=(obs_num, 2)) 
	#randomize the central positions of obstacles, avoid out of range of the array
	obs_bound = np.zeros((obs_num, 4)) 

	for i in range(obs_pos.shape[0]):
		a, b = obs_pos[i]
		random_style = random.randint(0, len(obs_style)-1)
		h, w = obs_style[random_style]
		# h, w = h//2, w//2
		maz[a:a+h, b:b+w] = 255
		obs_bound[i][0] = a
		obs_bound[i][1] = b
		obs_bound[i][2] = w
		obs_bound[i][3] = h

	def generate_sg(nc, space):
		s_g_mat = []

		for i in range(nc):
			# s_pos = np.random.randint(low=0, high=127, size=(2)) 
			# g_pos = np.random.randint(low=0, high=127, size=(2)) 
			

			while True:
				s_pos = np.random.randint(low=0, high=63, size=(2)) 
				g_pos = np.random.randint(low=0, high=63, size=(2)) 
				if space[s_pos[0], s_pos[1]] == 255 or space[g_pos[0], g_pos[1]] == 255 or np.sum(np.absolute(s_pos - g_pos)) < 100: 
					continue
				flag = True
				four_dir = np.array([[-1,-1],[-1,1],[1,-1],[1,1]])
				for direction in four_dir:
					tmp_s_pos = direction + s_pos
					tmp_g_pos = direction + g_pos
					if all(tmp_s_pos >=0) and space[tmp_s_pos[0], tmp_s_pos[1]] == 255:
						flag = False
					if all(tmp_g_pos >=0) and space[tmp_g_pos[0], tmp_g_pos[1]] == 255:
						flag = False
					# if all(tmp_s_pos >= 0) and all(tmp_g_pos >=0):
					# 	if space[tmp_s_pos[0], tmp_s_pos[1]] == 255 or space[tmp_g_pos[0], tmp_g_pos[1]] == 255:
					# 		Flag = False
				if flag:
					break

			s_g_mat.append(np.concatenate((s_pos, g_pos), axis=None))

		return np.stack(s_g_mat)

	save_maz(maz, maz_id)
	s_g_maz = generate_sg(num_config, maz)

	return maz, s_g_maz, obs_bound

def save_maz(maz, i):
	plt.imsave(op.join(pic_dir, '{}.png'.format(i)), maz)#, cmap='gray')

def generate_2D_dataset():
	total_maz = []
	total_config = []
	total_obs = []
	for i in range(100):
		print ('generating map', i)
		maz, s_g_confg, obs = generate_2D_map(64, 1000, i)
		total_maz.append(maz)
		total_config.append(s_g_confg)
		total_obs.append(obs)
	print ('total_config', total_config)
	total_config = np.stack(total_config)
	train_config = total_config[:,:800,:]
	test_seen_config = total_config[:,800:,:]

	test_unseen_config = []
	for i in range(100, 120):
		print ('generating map', i)
		maz, s_g_confg, obs = generate_2D_map(64, 400, i)
		total_maz.append(maz)
		test_unseen_config.append(s_g_confg)
		total_obs.append(obs)

	total_maz = np.stack(total_maz)
	test_unseen_config = np.stack(test_unseen_config)
	total_obs = np.stack(total_obs)

	print ('The statistics of the dataset...')
	print (total_maz.shape)
	print (train_config.shape)
	print (test_seen_config.shape)
	print (test_unseen_config.shape)
	print (total_obs.shape)

	# total_maz = total_maz.astype(np.uint8)
	# train_config = train_config.astype(np.uint8)
	# test_seen_config = test_seen_config.astype(np.uint8)
	# test_unseen_config = test_unseen_config.astype(np.uint8)
	total_maz = total_maz.astype(np.float32)
	train_config = train_config.astype(np.int32)
	test_seen_config = test_seen_config.astype(np.int32)
	test_unseen_config = test_unseen_config.astype(np.int32)
	total_obs = total_obs.astype(np.int32)

	print ('Saving the numpy array...')
	np.save(op.join(arr_dir, 'maz.npy'), total_maz)
	np.save(op.join(arr_dir, 'train_config.npy'), train_config)
	np.save(op.join(arr_dir, 'test_seen_config.npy'), test_seen_config)
	np.save(op.join(arr_dir, 'test_unseen_config.npy'), test_unseen_config)
	np.save(op.join(arr_dir, 'obs.npy'), total_obs)
	



def main():
	generate_2D_dataset()

if __name__ == '__main__':
	main()


