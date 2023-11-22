#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

import nnfs
from nnfs.datasets import spiral_data

from classes import Layer_Dense
from classes import Activation_ReLU, Activation_Softmax_Loss_CategoricalCrossEntropy
from classes import Optimizer_SGD, Optimizer_AdaGrad, Optimizer_RMSProp

nnfs.init()

# create dataset
X, y = spiral_data(samples = 100, classes = 3)

# create dense layer with 2 input features and 64 output values
dense1 = Layer_Dense(2, 64)

# create ReLU activation (to be used with dense layer):
activation1 = Activation_ReLU()

# create second dense layer with 64 input features (as we take output
# of previous layer here) and 3 output values (output values)
dense2 = Layer_Dense(64, 3)

# create softmax classifier's combined loss and activation
loss_activation = Activation_Softmax_Loss_CategoricalCrossEntropy()

# create optimizer
#optimizer = Optimizer_SGD(decay = 1e-3, momentum = 0.8)
#optimizer = Optimizer_AdaGrad(decay = 1e-4)
optimizer = Optimizer_RMSProp(learning_rate = 0.02, decay = 1e-5, rho = 0.999)

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

	# perform forward pass through the activation/loss function
	# takes the output of second dense layer here and returns loss
	loss = loss_activation.forward(dense2.output, y)

	# calculate accuracy from output of activation2 and targets
	# calculate values along first axis
	predictions = np.argmax(loss_activation.output, axis = 1)

	if len(y.shape) == 2:
		y = np.argmax(y, axis = 1)

	accuracy = np.mean(predictions == y)

	# print loss and accuracy values
	if not epoch % 500:
		print(f'epoch: {epoch}, acc: {accuracy:.3f}, loss: {loss:.3f}, lr: {optimizer.current_learning_rate:.6f}')

	# backward pass
	loss_activation.backward(loss_activation.output, y)
	dense2.backward(loss_activation.dinputs)
	activation1.backward(dense2.dinputs)
	dense1.backward(activation1.dinputs)

	# update weights and biases
	optimizer.pre_update_params()
	optimizer.update_params(dense1)
	optimizer.update_params(dense2)
	optimizer.post_update_params()

