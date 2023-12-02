import numpy as np


####################################################
# Softmax classifier                               #
# combined Softmax activation + cross-entropy loss #
# (for faster backward step)                       #
####################################################
class Activation_Softmax_Loss_CategoricalCrossEntropy:

	# backward pass
	def backward (self, dvalues, y_true):

		# number of samples
		samples = len(dvalues)

		# if labels are one-hot encoded, turn them into discrete values
		if len(y_true.shape) == 2:
			y_true = np.argmax(y_true, axis = 1)

		# copy so we can safely modify
		self.dinputs = dvalues.copy()

		# calc gradients
		self.dinputs[range(samples), y_true] -= 1

		# normalize gradients
		self.dinputs = self.dinputs / samples

