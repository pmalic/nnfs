#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

import nnfs
from nnfs.datasets import spiral_data

from classes import Layer_Dense, Layer_Dropout
from classes import Activation_ReLU, Activation_Softmax_Loss_CategoricalCrossEntropy
from classes import Optimizer_SGD, Optimizer_AdaGrad, Optimizer_RMSProp, Optimizer_Adam

nnfs.init()

# create dataset
X, y = spiral_data(samples = 1000, classes = 3)

# create dense layer with 2 input features and 512 output values
dense1 = Layer_Dense(2, 512, weight_regularizer_l2 = 5e-4,
					 									bias_regularizer_l2 = 5e-4)

# create ReLU activation (to be used with dense layer):
activation1 = Activation_ReLU()

# create dropout layer
dropout1 = Layer_Dropout(0.1)

# create second dense layer with 512 input features (as we take output
# of previous layer here) and 3 output values (output values)
dense2 = Layer_Dense(512, 3)

# create softmax classifier's combined loss and activation
loss_activation = Activation_Softmax_Loss_CategoricalCrossEntropy()

# create optimizer
#optimizer = Optimizer_SGD(decay = 1e-3, momentum = 0.8)
#optimizer = Optimizer_AdaGrad(decay = 1e-4)
#optimizer = Optimizer_RMSProp(learning_rate = 0.02, decay = 1e-5, rho = 0.999)
optimizer = Optimizer_Adam(learning_rate = 0.05, decay = 5e-5)

# train in loop
for epoch in range(10001):

	# perform forward pass of our training data through this layer
	dense1.forward(X)

	# perform forward pass through activation function
	# takes the output of first dense layer here
	activation1.forward(dense1.output)

	# perform forward pass through dropout layer
	dropout1.forward(activation1.output)

	# perform forward pass through second dense layer
	# takes outputs of activation function of first layer as inputs
	dense2.forward(dropout1.output)

	# perform forward pass through the activation/loss function
	# takes the output of second dense layer here and returns loss
	data_loss = loss_activation.forward(dense2.output, y)

	# calculate regularization penalty
	regularization_loss = loss_activation.loss.regularization_loss(dense1) + loss_activation.loss.regularization_loss(dense2)

	# calculate overall loss
	loss = data_loss + regularization_loss

	# calculate accuracy from output of activation2 and targets
	# calculate values along first axis
	predictions = np.argmax(loss_activation.output, axis = 1)

	if len(y.shape) == 2:
		y = np.argmax(y, axis = 1)

	accuracy = np.mean(predictions == y)

	# print loss and accuracy values
	if not epoch % 500:
		print(f'epoch: {epoch}, acc: {accuracy:.3f}, loss: {loss:.3f} (data_loss: {data_loss:.3f}, reg_loss: {regularization_loss:.3f}), lr: {optimizer.current_learning_rate:.6f}')

	# backward pass
	loss_activation.backward(loss_activation.output, y)
	dense2.backward(loss_activation.dinputs)
	dropout1.backward(dense2.dinputs)
	activation1.backward(dropout1.dinputs)
	dense1.backward(activation1.dinputs)

	# update weights and biases
	optimizer.pre_update_params()
	optimizer.update_params(dense1)
	optimizer.update_params(dense2)
	optimizer.post_update_params()


# Validate the model

# Create test dataset
X_test, y_test = spiral_data(samples = 100, classes = 3)

# Perform a forward pass of our testing data through this layer
dense1.forward(X_test)

# Perform a forward pass through activation function
# takes the output of first dense layer here
activation1.forward(dense1.output)

# Perform a forward pass through second Dense layer
# takes outputs of activation function of first layer as inputs
dense2.forward(activation1.output)

# Perform a forward pass through the activation/loss function
# takes the output of second dense layer here and returns loss
loss = loss_activation.forward(dense2.output, y_test)

# Calculate accuracy from output of activation2 and targets
# # calculate values along first axis
predictions = np.argmax(loss_activation.output, axis = 1)

if len(y_test.shape) == 2:
	y_test = np.argmax(y_test, axis = 1)

accuracy = np.mean(predictions == y_test)

print(f'validation, acc: {accuracy:.3f}, loss: {loss:.3f}')
