#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

import nnfs
from nnfs.datasets import spiral_data

from model import Model
from layer import Layer_Dense, Layer_Dropout
from activation import Activation_ReLU, Activation_Softmax
from optimizer import Optimizer_Adam
from loss import Loss_CategoricalCrossEntropy
from accuracy import Accuracy_Categorical

nnfs.init()

# create dataset
X, y = spiral_data(samples = 1000, classes = 3)
X_test, y_test = spiral_data(samples = 100, classes = 3)

# instantiate the model
model = Model()

# add layers
model.add(Layer_Dense(2, 512, weight_regularizer_l2 = 5e-4,
                      		 bias_regularizer_l2 = 5e-4))
model.add(Activation_ReLU())
model.add(Layer_Dropout(0.1))
model.add(Layer_Dense(512, 3))
model.add(Activation_Softmax())

# set loss, optimizer and accuracy objects
model.set(
    loss = Loss_CategoricalCrossEntropy(),
    optimizer = Optimizer_Adam(learning_rate = 0.05, decay = 5e-5),
	accuracy = Accuracy_Categorical()
)

# finalize the model
model.finalize()

# train the model
model.train(X, y, validation_data = (X_test, y_test), epochs = 10000, print_every = 500)

