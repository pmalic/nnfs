#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

from model import Model
from layer import Layer_Dense
from activation import Activation_ReLU, Activation_Softmax
from optimizer import Optimizer_Adam
from loss import Loss_CategoricalCrossEntropy
from accuracy import Accuracy_Categorical

from dataset import create_data_mnist

# create dataset
X, y, X_test, y_test = create_data_mnist('fashion_mnist_images')

# shuffle the training dataset
keys = np.array(range(X.shape[0]))
np.random.shuffle(keys)

X = X[keys]
y = y[keys]

# reshape and scale samples
X = (X.reshape(X.shape[0], -1).astype(np.float32) - 127.5) / 127.5
X_test = (X_test.reshape(X_test.shape[0], -1).astype(np.float32) - 127.5) / 127.5

# instantiate the model
model = Model()

# add layers
model.add(Layer_Dense(X.shape[1], 128))
model.add(Activation_ReLU())
model.add(Layer_Dense(128, 128))
model.add(Activation_ReLU())
model.add(Layer_Dense(128, 10))
model.add(Activation_Softmax())

# set loss, optimizer and accuracy objects
model.set(
    loss = Loss_CategoricalCrossEntropy(),
    optimizer = Optimizer_Adam(decay = 1e-3),
	accuracy = Accuracy_Categorical()
)

# finalize the model
model.finalize()

# train the model
model.train(X, y, validation_data = (X_test, y_test), epochs = 10, batch_size = 128, print_every = 100)

# evaluate the model
model.evaluate(X, y)

