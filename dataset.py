import numpy as np
import cv2
import os


# loads a MNIST dataset
def load_mnist_dataset (dataset, path):

	# scan all the directories and create a list of labels
	labels = os.listdir(os.path.join(path, dataset))

	# create lists for samples and labels
	X = []
	y = []

	# for each label folder
	for label in labels:

		# and for each image in given folder
		for file in os.listdir(os.path.join(path, dataset, label)):

			# read the image
			image = cv2.imread(os.path.join(path, dataset, label, file), cv2.IMREAD_UNCHANGED)

			# and append it and a label to the lists
			X.append(image)
			y.append(label)
		#}

	# convert the data to proper numpy arrays and return
	return np.array(X), np.array(y).astype('uint8')


# MNIST dataset (train + test)
def create_data_mnist (path):

	# load both sets separately
	X, y = load_mnist_dataset('train', path)
	X_test, y_test = load_mnist_dataset('test', path)

	# and return all the data
	return X, y, X_test, y_test

