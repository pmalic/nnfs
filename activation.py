import numpy as np


###################
# ReLU activation #
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
# Softmax activation #
######################
class Activation_Softmax:

	# forward pass
	def forward (self, inputs):

		self.inputs = inputs

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


#######################
# Signmoid activation #
#######################
class Activation_Sigmoid:

	# forward pass
	def forward (self, inputs):

		# save input and calculate/save output
		# of sigmoid function
		self.inputs = inputs
		self.output = 1 / (1 + np.exp(- inputs))


	def backward (self, dvalues):

		# derivative - calculates from output of sigmoid function
		self.dinputs = dvalues * self.output * (1 - self.output)


#####################
# Linear activation #
#####################
class Activation_Linear:

	# forward pass
	def forward (self, inputs):

		# just remember values
		self.inputs = inputs
		self.output = inputs


	# backward pass
	def backward (self, dvalues):

		# derivative is 1, 1 * dvalues = dvalues - chain rule
		self.dinputs = dvalues.copy()

