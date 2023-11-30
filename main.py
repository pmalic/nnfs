#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

import nnfs
from nnfs.datasets import spiral_data

from classes import Layer_Dense, Layer_Dropout
from classes import Activation_ReLU, Activation_Softmax_Loss_CategoricalCrossEntropy, Activation_Sigmoid
from classes import Loss_BinaryCrossEntropy
from classes import Optimizer_SGD, Optimizer_AdaGrad, Optimizer_RMSProp, Optimizer_Adam

nnfs.init()

# create dataset
X, y = spiral_data(samples = 100, classes = 2)

# reshape labels to be a list of lists
# inner list contains one output (either 0 or 1)
# per each output neuron, 1 in this case
y = y.reshape(-1, 1)

# create dense layer with 2 input features and 64 output values
dense1 = Layer_Dense(2, 64, weight_regularizer_l2 = 5e-4,
					 									bias_regularizer_l2 = 5e-4)

# create ReLU activation (to be used with dense layer):
activation1 = Activation_ReLU()

# create second dense layer with 64 input features (as we take output
# of previous layer here) and 1 output value
dense2 = Layer_Dense(64, 1)

# create sigmoid activation
activation2 = Activation_Sigmoid()

# create loss function
loss_function = Loss_BinaryCrossEntropy()

# create optimizer
optimizer = Optimizer_Adam(decay = 5e-7)

# train in loop
for epoch in range(10001):

	# perform forward pass of our training data through this layer
	dense1.forward(X)

	# perform forward pass through activation function
	# takes the output of first dense layer here
	activation1.forward(dense1.output)

	# perform forward pass through second dense layer
	# takes outputs of activation function of first layer as inputs
	dense2.forward(activation1.output)

	# perform a forward pass through the activation function
	# takes the output of second dense layer here
	activation2.forward(dense2.output)

	# perform forward pass through the activation/loss function
	# takes the output of second dense layer here and returns loss
	data_loss = loss_function.calculate(activation2.output, y)

	# calculate regularization penalty
	regularization_loss = loss_function.regularization_loss(dense1) + loss_function.regularization_loss(dense2)

	# calculate overall loss
	loss = data_loss + regularization_loss

	# calculate accuracy from output of activation2 and targets
	# part in the brackets returns a binary mask - array consisting
	# of True/False values, multiplying it by 1 changes it into array
	# of 1s and 0s
	predictions = (activation2.output > 0.5) * 1
	accuracy = np.mean(predictions == y)

	# print loss and accuracy values
	if not epoch % 500:
		print(f'epoch: {epoch}, acc: {accuracy:.3f}, loss: {loss:.3f} (data_loss: {data_loss:.3f}, reg_loss: {regularization_loss:.3f}), lr: {optimizer.current_learning_rate:.6f}')

	# backward pass
	loss_function.backward(activation2.output, y)
	activation2.backward(loss_function.dinputs)
	dense2.backward(activation2.dinputs)
	activation1.backward(dense2.dinputs)
	dense1.backward(activation1.dinputs)

	# update weights and biases
	optimizer.pre_update_params()
	optimizer.update_params(dense1)
	optimizer.update_params(dense2)
	optimizer.post_update_params()


# validate the model

# create test dataset
X_test, y_test = spiral_data(samples = 100, classes = 2)

# reshape labels to be a list of lists
# inner list contains one output (either 0 or 1)
# per each output neuron, 1 in this case
y_test = y_test.reshape(-1, 1)

# perform a forward pass of our testing data through this layer
dense1.forward(X_test)

# perform a forward pass through activation function
# takes the output of first dense layer here
activation1.forward(dense1.output)

# perform a forward pass through second Dense layer
# takes outputs of activation function of first layer as inputs
dense2.forward(activation1.output)

# perform a forward pass through the activation function
# takes the output of second dense layer here
activation2.forward(dense2.output)

# calculate the data loss
loss = loss_function.calculate(activation2.output, y_test)

# calculate accuracy from output of activation2 and targets
# part in the brackets returns a binary mask - array consisting
# true/false values, multiplying it by 1 changes it into array
# of 1s and 0s
predictions = (activation2.output > 0.5) * 1
accuracy = np.mean(predictions == y_test)

print(f'validation, acc: {accuracy:.3f}, loss: {loss:.3f}')
