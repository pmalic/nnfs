import numpy as np

from activation import Activation_Softmax
from loss import Loss_CategoricalCrossEntropy


###########################################
# Softmax activation + cross-entropy loss #
# (for faster backward step)              #
###########################################
class Activation_Softmax_Loss_CategoricalCrossEntropy:

	def __init__ (self):

		self.activation = Activation_Softmax()
		self.loss = Loss_CategoricalCrossEntropy()


	# forward pass
	def forward (self, inputs, y_true):

		# output layer's activation function
		self.activation.forward(inputs)

		# set the output
		self.output = self.activation.output

		# calculate and return loss value
		return self.loss.calculate(self.output, y_true)


	# backward pass
	def backward (self, dvalues, y_true):

		samples = len(dvalues)

		# un-hot-encode
		if len(y_true.shape) == 2:
			y_true = np.argmax(y_true, axis = 1)

		self.dinputs = dvalues.copy()

		# calc gradients
		self.dinputs[range(samples), y_true] -= 1

		# normalize gradients
		self.dinputs = self.dinputs / samples

