import numpy as np


###############
# Dense layer #
###############
class Layer_Dense:

	def __init__ (self, n_inputs, n_neurons, weight_regularizer_l1 = 0, weight_regularizer_l2 = 0, bias_regularizer_l1 = 0, bias_regularizer_l2 = 0):

		# initialize weights and biases
		self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
		self.biases = np.zeros((1, n_neurons))

		# set regularization strength
		self.weight_regularizer_l1 = weight_regularizer_l1
		self.weight_regularizer_l2 = weight_regularizer_l2
		self.bias_regularizer_l1 = bias_regularizer_l1
		self.bias_regularizer_l2 = bias_regularizer_l2


	# forward pass
	def forward (self, inputs):

		self.inputs = inputs
		self.output = np.dot(inputs, self.weights) + self.biases


	# backward pass
	def backward (self, dvalues):

		# gradients on params
		self.dweights = np.dot(self.inputs.T, dvalues)
		self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

		# gradients on regularization
		# L1 on weights
		if self.weight_regularizer_l1 > 0:

				dL1 = np.ones_like(self.weights)
				dL1[self.weights < 0] = -1
				self.dweights += self.weight_regularizer_l1 * dL1

		# L2 on weights
		if self.weight_regularizer_l2 > 0:
				self.dweights += 2 * self.weight_regularizer_l2 * self.weights

		# L1 on biases
		if self.bias_regularizer_l1 > 0:

				dL1 = np.ones_like(self.biases)
				dL1[self.biases < 0] = -1
				self.dbiases += self.bias_regularizer_l1 * dL1

		# L2 on biases
		if self.bias_regularizer_l2 > 0:
				self.dbiases += 2 * self.bias_regularizer_l2 * self.biases

		# gradient on values
		self.dinputs = np.dot(dvalues, self.weights.T)


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


#################
# SGD optimizer #
#################
class Optimizer_SGD:

	# initialize optimizer - set settings
	# learning rate of 1. is default for this optimizer
	def __init__ (self, learning_rate = 1., decay = 0., momentum = 0.):

		self.learning_rate = learning_rate
		self.current_learning_rate = learning_rate
		self.decay = decay
		self.momentum = momentum

		self.iterations = 0


	# call once before any parameter updates
	def pre_update_params (self):

		if self.decay:
			self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))


	# update params
	def update_params (self, layer):

		# if we use momentum
		if self.momentum:

			# if layer doesn't contain momentum arrays,
			# create them filled with zeros
			if not hasattr(layer, 'weight_momentums'):
				layer.weight_momentums = np.zeros_like(layer.weights)
				layer.bias_momentums = np.zeros_like(layer.biases)

			# build weight updates with momentum - take previous
			# updates multiplied by retain factor and update with
			# current gradients
			weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
			layer.weight_momentums = weight_updates

			# build bias updates
			bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
			layer.bias_momentums = bias_updates

		# vanilla SGD updates (as before momentum update)
		else:
			weight_updates = - self.current_learning_rate * layer.dweights
			bias_updates = - self.current_learning_rate * layer.dbiases

		# update weights and biases using either
		# vanilla or momentum updates
		layer.weights += weight_updates
		layer.biases += bias_updates


	# call once after any parameter updates
	def post_update_params (self):

		self.iterations += 1


#####################
# AdaGrad optimizer #
#####################
class Optimizer_AdaGrad:

	# initialize optimizer - set settings
	def __init__ (self, learning_rate = 1., decay = 0., epsilon = 1e-7):

		self.learning_rate = learning_rate
		self.current_learning_rate = learning_rate
		self.decay = decay
		self.epsilon = epsilon

		self.iterations = 0


	# call once before any parameter updates
	def pre_update_params (self):

		if self.decay:
			self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))


	# update params
	def update_params (self, layer):

		# if layer doesn't contain cache arrays,
		# create them filled with zeros
		if not hasattr(layer, 'weight_cache'):
			layer.weight_cache = np.zeros_like(layer.weights)
			layer.bias_cache = np.zeros_like(layer.biases)

		# update cache with squared current gradients
		layer.weight_cache += layer.dweights ** 2
		layer.bias_cache += layer.dbiases ** 2

		# vanilla SGD parameter update + normalization
		# with square rooted cache
		layer.weights += - self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
		layer.biases += - self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)


	# call once after any parameter updates
	def post_update_params (self):

		self.iterations += 1


#####################
# RMSProp optimizer #
#####################
class Optimizer_RMSProp:

	# initialize optimizer - set settings
	def __init__ (self, learning_rate = .001, decay = 0., epsilon = 1e-7, rho = 0.9):

		self.learning_rate = learning_rate
		self.current_learning_rate = learning_rate
		self.decay = decay
		self.epsilon = epsilon
		self.rho = rho

		self.iterations = 0


	# call once before any parameter updates
	def pre_update_params (self):

		if self.decay:
			self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))


	# update params
	def update_params (self, layer):

		# if layer doesn't contain cache arrays,
		# create them filled with zeros
		if not hasattr(layer, 'weight_cache'):
			layer.weight_cache = np.zeros_like(layer.weights)
			layer.bias_cache = np.zeros_like(layer.biases)

		# update cache with squared current gradients
		layer.weight_cache = self.rho * layer.weight_cache + (1 - self.rho) * layer.dweights ** 2
		layer.bias_cache = self.rho * layer.bias_cache + (1 - self.rho) * layer.dbiases ** 2

		# vanilla SGD parameter update + normalization
		# with square rooted cache
		layer.weights += - self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
		layer.biases += - self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)


	# call once after any parameter updates
	def post_update_params (self):

		self.iterations += 1


##################
# Adam optimizer #
##################
class Optimizer_Adam:

	# initialize optimizer - set settings
	def __init__ (self, learning_rate = .001, decay = 0., epsilon = 1e-7, beta_1 = 0.9, beta_2 = 0.999):

		self.learning_rate = learning_rate
		self.current_learning_rate = learning_rate
		self.decay = decay
		self.epsilon = epsilon
		self.beta_1 = beta_1
		self.beta_2 = beta_2

		self.iterations = 0


	# call once before any parameter updates
	def pre_update_params (self):

		if self.decay:
			self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))


	# update params
	def update_params (self, layer):

		# if layer doesn't contain cache arrays,
		# create them filled with zeros
		if not hasattr(layer, 'weight_cache'):

			layer.weight_momentums = np.zeros_like(layer.weights)
			layer.weight_cache = np.zeros_like(layer.weights)

			layer.bias_momentums = np.zeros_like(layer.biases)
			layer.bias_cache = np.zeros_like(layer.biases)

		# update momentum with current gradients
		layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
		layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases

		# get corrected momentum
		# self.iteration is 0 at first pass
		# and we need to start with 1 here
		weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
		bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))

		# update cache with squared current gradients
		layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights ** 2
		layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases ** 2

		# get corrected cache
		weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
		bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))


		# vanilla SGD parameter update + normalization
		# with square rooted cache
		layer.weights += - self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
		layer.biases += - self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)


	# call once after any parameter updates
	def post_update_params (self):

		self.iterations += 1


#############
# Base loss #
#############
class Loss:

	# regularization loss calculation
	def regularization_loss (self, layer):

		# 0 by default
		regularization_loss = 0

		# L1 regularization - weights
		# calculate only when factor greater than 0
		if layer.weight_regularizer_l1 > 0:
			regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))

		# L2 regularization - weights
		if layer.weight_regularizer_l2 > 0:
			regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights ** 2)

		# L1 regularization - biases
		# calculate only when factor greater than 0
		if layer.bias_regularizer_l1 > 0:
			regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))

		# L2 regularization - biases
		if layer.bias_regularizer_l2 > 0:
			regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases ** 2)

		return regularization_loss


	# calculates the data and regularization losses
	# given model output and ground truth values
	def calculate (self, output, y):

		# calculate sample losses
		sample_losses = self.forward(output, y)

		# calculate mean loss
		data_loss = np.mean(sample_losses)

		return data_loss


##################################
# Categorical cross-entropy loss #
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

