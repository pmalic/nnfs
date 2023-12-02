import numpy as np


#################
# Base accuracy #
#################
class Accuracy:

	# calculates an accuracy
	# given predictions and ground truth values
	def calculate (self, predictions, y):

		# get comparison results
		comparisons = self.compare(predictions, y)

		# calculate an accuracy
		accuracy = np.mean(comparisons)

		return accuracy


#################################
# Accuracy for regression model #
#################################
class Accuracy_Regression (Accuracy):

	def __init__ (self):

		# create precision property
		self.precision = None


	# calculates precision value
	# based on passed in ground truth values
	def init (self, y, reinit = False):

		if self.precision is None or reinit:
			self.precision = np.std(y) / 250


	# compares predictions to the ground truth values
	def compare (self, predictions, y):

		return np.absolute(predictions - y) < self.precision


#####################################
# Accuracy for classification model #
#####################################
class Accuracy_Categorical (Accuracy):

	def __init__ (self, *, binary = False):

		# binary mode?
		self.binary = binary


	# no initialization is needed
	def init (self, y):

		pass


	# compares predictions to the ground truth values
	def compare (self, predictions, y):

		if not self.binary and len(y.shape) == 2:
			y = np.argmax(y, axis = 1)

		return predictions == y

