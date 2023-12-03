import numpy as np


#############
# Base loss #
#############
class Loss:

	# regularization loss calculation
	def regularization_loss (self):

		# 0 by default
		regularization_loss = 0

		for layer in self.trainable_layers:

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
		#}

		return regularization_loss


	# set/remember trainable layers
	def remember_trainable_layers (self, trainable_layers):

		self.trainable_layers = trainable_layers


	# calculates the data and regularization losses
	# given model output and ground truth values
	def calculate (self, output, y, *, include_regularization = False):

		# calculate sample losses
		sample_losses = self.forward(output, y)

		# calculate mean loss
		data_loss = np.mean(sample_losses)

		# add accumulated sum of losses and sample count
		self.accumulated_sum += np.sum(sample_losses)
		self.accumulated_count += len(sample_losses)

		# if just data loss - return it
		if not include_regularization:
			return data_loss

		# return the data and regularization losses
		return data_loss, self.regularization_loss()


	# calculates accumulated loss
	def calculate_accumulated (self, *, include_regularization = False):

		# calculate mean loss
		data_loss = self.accumulated_sum / self.accumulated_count

		# if just data loss - return it
		if not include_regularization:
			return data_loss

		# return the data and regularization losses
		return data_loss, self.regularization_loss()


	# resets accumulated loss
	def new_pass (self):

		self.accumulated_sum = 0
		self.accumulated_count = 0


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


#############################
# Binary cross-entropy loss #
#############################
class Loss_BinaryCrossEntropy (Loss):

	# forward pass
	def forward (self, y_pred, y_true):

		# clip data to prevent division by 0
		# clip both sides to not drag mean towards any value
		y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

		# calculate sample-wise loss
		sample_losses = - (y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
		sample_losses = np.mean(sample_losses, axis = -1)

		return sample_losses


	# backward pass
	def backward (self, dvalues, y_true):

		# number of samples
		samples = len(dvalues)

		# number of outputs in every sample
		# we'll use the first sample to count them
		outputs = len(dvalues[0])

		# clip data to prevent division by 0
		# clip both sides to not drag mean towards any value
		clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)

		# calc gradient
		self.dinputs = - (y_true / clipped_dvalues - (1 - y_true) / (1 - clipped_dvalues)) / outputs

		# normalize gradient
		self.dinputs = self.dinputs / samples


###########################
# Mean Squared Error loss #
###########################
class Loss_MeanSquaredError (Loss): # L2 loss

	# forward pass
	def forward (self, y_pred, y_true):

		# calculate loss
		sample_losses = np.mean((y_true - y_pred) ** 2, axis = -1)

		return sample_losses


	# backward pass
	def backward (self, dvalues, y_true):

		# number of samples
		samples = len(dvalues)

		# number of outputs in every sample
		# we'll use the first sample to count them
		outputs = len(dvalues[0])

		# gradient on values
		self.dinputs = -2 * (y_true - dvalues) / outputs

		# normalize gradient
		self.dinputs = self.dinputs / samples


############################
# Mean Absolute Error loss #
############################
class Loss_MeanAbsoluteError (Loss): # L1 loss

	# forward pass
	def forward (self, y_pred, y_true):

		# calculate loss
		sample_losses = np.mean(np.abs(y_true - y_pred), axis = -1)

		return sample_losses


	# backward pass
	def backward (self, dvalues, y_true):

		# number of samples
		samples = len(dvalues)

		# number of outputs in every sample
		# we'll use the first sample to count them
		outputs = len(dvalues[0])

		# calculate gradient
		self.dinputs = np.sign(y_true - dvalues) / outputs

		# normalize gradient
		self.dinputs = self.dinputs / samples

