import numpy as np


###############
# dense layer #
###############
class Layer_Dense:

	def __init__ (self, n_inputs, n_neurons):

		self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
		self.biases = np.zeros((1, n_neurons))

	# forward pass
	def forward (self, inputs):

		self.inputs = inputs
		self.output = np.dot(inputs, self.weights) + self.biases

	# backward pass
	def backward (self, dvalues):

		# gradients on params
		self.dweights = np.dot(self.inputs.T, dvalues)
		self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

		# gradient on values
		self.dinputs = np.dot(dvalues, self.weights.T)


###################
# relu activation #
###################
class Activation_ReLU:

	# forward pass
	def forward (self, inputs):

		self.inputs = inputs
		self.output = np.maximum(0, inputs)

	# backward pass
	def backward (self, dvalues):

		self.dinputs = dvalues.copy()
		self.dinputs[self.inputs <= 0] = 0


######################
# softmax activation #
######################
class Activation_Softmax:

	# forward pass
	def forward (self, inputs):

		exp_values = np.exp(inputs - np.max(inputs, axis = 1, keepdims = True))
		probabilities = exp_values / np.sum(exp_values, axis = 1, keepdims = True)

		self.output = probabilities

	# backward pass
	def backward (self, dvalues):

		self.dinputs = np.empty_like(dvalues)

		for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):

			single_output = single_output.reshape(-1, 1)

			jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)

			self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)


#############
# base loss #
#############
class Loss:

	def calculate (self, output, y):

		sample_losses = self.forward(output, y)

		data_loss = np.mean(sample_losses)

		return data_loss


##################################
# categorical cross-entropy loss #
##################################
class Loss_CategoricalCrossEntropy (Loss):

	# forward pass
	def forward (self, y_pred, y_true):

		samples = len(y_pred)

		y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

		if len(y_true.shape) == 1:

			correct_confidences = y_pred_clipped[ range(samples), y_true ]

		elif len(y_true.shape) == 2:

			correct_confidences = np.sum(y_pred_clipped * y_true, axis = 1)

		negative_log_likelihoods = - np.log(correct_confidences)

		return negative_log_likelihoods

	# backward pass
	def backward (self, dvalues, y_true):

		samples = len(dvalues)

		labels = len(dvalues[0])

		if len(y_true.shape) == 1:
			y_true = np.eye(labels)[y_true]

		# calc gradient
		self.dinputs = - y_true / dvalues

		# normalize gradient
		self.dinputs = self.dinputs / samples


###########################################
# softmax activation + cross-entropy loss #
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
			y_true = np.argmax(y_true, axis=1)

		self.dinputs = dvalues.copy()

		# calc gradients
		self.dinputs[range(samples), y_true] -= 1

		# normalize gradients
		self.dinputs = self.dinputs / samples

