# Tools to augment images and generate batches with a generator

import cv2
import os
import numpy as np
import matplotlib.image as mpimg

IMAGE_HEIGHT_ORIGINAL, IMAGE_WIDTH_ORIGINAL, IMAGE_CHANNELS_ORIGINAL = 160, 320, 3
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)


def load_image(data_dir, image_file):
	"""
	Load RGB image 
	Arguments:
		@data_dir: path where image is stored
		@image_file: name of image file
	Returns:
		Loaded image
	"""
	return mpimg.imread(os.path.join(data_dir, image_file.strip()))


def crop(image):
	"""
	Crop image (In behavioral cloning project this helps to remove the sky and front of car)
	Arguments:
		@image: Image to crop
	Returns:
		Cropped image
	"""
	return image[60:-25, :, :]
	#return image[120:-50, :, :]


def resize(image):
	"""
	Resize the image to the input shape used by the network model
	Arguments:
		@image: Image
	Returns:
		Image resized to the CNN input shape
	"""
	return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)


def rgb2yuv(image):
	"""
	Convert the image from RGB to YUV (Taken the approach from NVIDA paper)
	Arguments:
		@image: Image in RGB color space
	Returns:
		Image in YUV space
	"""
	return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)


def preprocess(image):
	"""
	Use all preprocess functions in here
	Arguments:
		@image: Image
	Returns:
		Procesed image
	"""
	image = crop(image)
	image = resize(image)
	image = rgb2yuv(image)
	return image


def choose_image(data_dir, center, left, right, steering_angle):
	"""
	Randomly choose a center, left or right image and adjust the steering angle
	Arguments:
		@data_dir: path where the image is stored
		@center: center file name
		@left: left file name
		@right: right file name
		@steering_angle: steering angle at the moment the image was logged
	Returns:
		Image and adjusted steering angle
	"""
	LEFT = 0
	RIGHT = 1
	STEERING_ANGLE_FUDGE_FACTOR = 0.2
	random_choice = np.random.choice(3)
	if random_choice == LEFT:
		steering_angle+=STEERING_ANGLE_FUDGE_FACTOR
		return load_image(data_dir, left), steering_angle
	elif random_choice == RIGHT:
		steering_angle-=STEERING_ANGLE_FUDGE_FACTOR
		return load_image(data_dir, right), steering_angle
	return load_image(data_dir, center), steering_angle


def random_flip(image, steering_angle):
	"""
	Randomly flip images from the horizontal (mirror images) and adjust the steering angle
	Arguments:
		@image: Image to probably flip
		@steering_angle: steering angle of the image
	Returns:
		Image maybe flipped with the corresponding steering angle
	"""
	if np.random.rand() < 0.5:
		image = cv2.flip(image, 1)
		steering_angle = -steering_angle # the steering angle of a flipped image is the original angle times -1 because it is the opposite as the original (mirror)
	return image, steering_angle


def random_translate(image, steering_angle, range_x, range_y):
	"""
	Randomly translate the image vertically and horizontally
	Arguments:
		@image: Image
		@steering_angle: steering angle associated to the image
		@range_x: translation associated to x axis
		@range_y: translation associated to y axis
	Returns:
		Return the translated image and the steering angle  
	"""
	trans_x = range_x * (np.random.rand() - 0.5)
	trans_y = range_y * (np.random.rand() - 0.5)
	steering_angle += trans_x * 0.002
	homogeneous_trans_2Dmatrix = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
	height, width = image.shape[:2]
	image = cv2.warpAffine(image, homogeneous_trans_2Dmatrix, (width, height))
	return image, steering_angle


def random_shadow(image):
	"""
	Randomly adds shadows to image
	Arguments:
		@image: Image
	Returns:
		Image with random shadow
	"""
	# Form a line as baseline to determine where to add a shadow
	# Points (x1,y1) and (x2,y2)
	IMAGE_WIDTH_SHADOW =image.shape[1]
	IMAGE_HEIGHT_SHADOW = image.shape[0]
	x1, y1 = IMAGE_WIDTH_SHADOW * np.random.rand(), 0
	x2, y2 = IMAGE_WIDTH_SHADOW * np.random.rand(), IMAGE_HEIGHT_SHADOW
	# Frame of whole points in the image
	xm, ym = np.mgrid[0:IMAGE_HEIGHT_SHADOW, 0:IMAGE_WIDTH_SHADOW]

	# To give shadow to some pixels, one needs to create a mask
	# the line above helps us do that and we set 1 or 0 to pixels
	# NOTE: Remember coordinates are up side down in an image
	# (ym - y1)/(xm - x1) > (y2 - y1)/(x2 - x1) 
	# if x2 = x1 then x2 - x1 = 0 and we can have a division by zero, so better write it:
	# (ym - y1)*(x2 - x1) - (y2 - y1)*(xm - x1) > 0
	mask = np.zeros_like(image[:,:,1])
	mask[ ((ym - y1)*(x2 - x1) - (y2 - y1)*(xm - x1)) > 0] = 1
	# choose the side that will have a shadow and adjust saturation
	cond = mask == np.random.randint(2)
	s_ratio = np.random.uniform(low=0.2, high=0.5)

	# adjust Saturation in HLS(Hue, Light, Saturation)
	hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
	hls[:,:,1][cond] = hls[:,:,1][cond] * s_ratio

	return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)	


def random_brightness(image):
	"""
	Randomly adjust brightness of the image using HSV space
	Arguments:	
		@image: Image
	Returns:
		Image w same or different brightness
	"""
	# HSV (Hue, Saturation, Value) - Value is Brightness
	hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
	ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
	hsv[:,:,2] = hsv[:,:,2] * ratio
	return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def augment(data_dir, center, left, right, steering_angle, range_x=100, range_y=100):
	"""
	Augments images and adjust steering angles 
	"""
	image, steering_angle = choose_image(data_dir, center, left, right, steering_angle)
	#image = crop(image)
	#image = resize(image)
	image, steering_angle = random_flip(image, steering_angle)
	image, steering_angle = random_translate(image, steering_angle, range_x, range_y)
	image = random_shadow(image)
	image = random_brightness(image)
	return image, steering_angle

def batch_generator(data_dir, image_paths, steering_angles, batch_size, is_training):
	"""
	Generator of images and steers batches
	"""
	images = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
	steers = np.empty(batch_size)
	while True:
		i = 0
		for index in np.random.permutation(image_paths.shape[0]):
			center, left, right = image_paths[index]
			steering_angle = steering_angles[index]
			# augmentation 
			if is_training and np.random.rand() < 0.6:
				image, steering_angle = augment(data_dir, center, left, right, steering_angle)
				#image = rgb2yuv(image)
			else:
				image = load_image(data_dir, center)
				#image = preprocess(image)#fix
			# preprocess image and then add to batch
			images[i] = preprocess(image)
			#images[i] = image #augment() crops and resizes images before actual augmentation 
			steers[i] = steering_angle
			i+=1
			if i == batch_size:
				break
		yield images, steers

 
