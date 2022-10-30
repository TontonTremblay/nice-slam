import cv2
import glob
import numpy as np
import random
import imageio

for name in glob.glob('Datasets/Demo/frames/color_/*.jpg'):
	im = cv2.imread(name)
	im = cv2.resize(im,(640, 480))
	black = np.zeros(im.shape)
	s_x = random.randint(40,150)
	s_y = random.randint(40,150)
	# s_x = 0
	# s_y = 0
	# black = cv2.resize(black,(640, 480))
	black[s_x:-(s_x+1),s_y:-(s_y+1),:] = im[s_x:-(s_x+1),s_y:-(s_y+1),:]
	cv2.imwrite(name.replace("_",''),black)

	# update the depth
	print(name.replace("color_","depth_"))
	dname = name.replace("color_","depth_").replace("jpg",'png')
	depth_data = cv2.imread(dname, cv2.IMREAD_UNCHANGED)

	# depth_data[s_x:-(s_x+1),s_y:-(s_y+1)] = 0
	# black = np.zeros(depth_data.shape)
	black = np.ones(depth_data.shape)*np.iinfo(depth_data.dtype).max
	black = np.ones(depth_data.shape)*-1

	black[s_x:-(s_x+1),s_y:-(s_y+1)] = depth_data[s_x:-(s_x+1),s_y:-(s_y+1)]
	print("depthdata",depth_data.min(),depth_data.max())
	print('to_output',black.min(),black.max())
	output_name = dname.replace("_","").replace('png','exr')
	# output_name = dname.replace("_","")
	# cv2.imwrite(output_name,black)

	imageio.imwrite(output_name, black.astype("float32"))

	# b = cv2.imread(output_name, cv2.IMREAD_UNCHANGED)
	# print(b.min())
	# print("reloaded",b.min(),b.max())

	# raise()