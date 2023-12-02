#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

import nnfs
from nnfs.datasets import sine_data

from model import Model
from layer import Layer_Dense
from activation import Activation_ReLU, Activation_Linear
from loss import Loss_MeanSquaredError
from accuracy import Accuracy_Regression
from optimizer import Optimizer_Adam

nnfs.init()

# create dataset
X, y = sine_data()

# instantiate the model
model = Model()

# add layers
model.add(Layer_Dense(1, 64))
model.add(Activation_ReLU())
model.add(Layer_Dense(64, 64))
model.add(Activation_ReLU())
model.add(Layer_Dense(64, 1))
model.add(Activation_Linear())

# set loss, optimizer and accuracy objects
model.set(
    loss = Loss_MeanSquaredError(),
    optimizer = Optimizer_Adam(learning_rate = 0.005, decay = 1e-3),
	accuracy = Accuracy_Regression()
)

# finalize the model
model.finalize()

# train the model
model.train(X, y, epochs = 10000, print_every = 500, validation_data = sine_data())

