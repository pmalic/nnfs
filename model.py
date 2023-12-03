from layer import Layer_Input
from activation import Activation_Softmax
from loss import Loss_CategoricalCrossEntropy
from combined import Activation_Softmax_Loss_CategoricalCrossEntropy


#########
# Model #
#########
class Model:

	def __init__ (self):

		# create a list of network objects
		self.layers = []

		# softmax classifier's output object
		self.softmax_classifier_output = None


	# add objects to the model
	def add (self, layer):

		self.layers.append(layer)


	# set loss and optimizer
	def set (self, *, loss, optimizer, accuracy):

		self.loss = loss
		self.optimizer = optimizer
		self.accuracy = accuracy


	# finalize the model
	def finalize (self):

		# create and set the input layer
		self.input_layer = Layer_Input()

		# count all the objects
		layer_count = len(self.layers)

		# initialize a list containing trainable layers
		self.trainable_layers = []

		# iterate the objects
		for i in range(layer_count):

			# if it's the first layer,
			# the previous layer object is the input layer
			if i == 0:

				self.layers[i].prev = self.input_layer
				self.layers[i].next = self.layers[i + 1]

			# all layers except for the first and the last
			elif i < layer_count - 1:

				self.layers[i].prev = self.layers[i - 1]
				self.layers[i].next = self.layers[i + 1]

			# the last layer - the next object is the loss
			else:
				self.layers[i].prev = self.layers[i - 1]
				self.layers[i].next = self.loss
				self.output_layer_activation = self.layers[i]
			#}

			# if the layer contains an attribute called "weights",
			# it's a trainable layer - add it to the list of trainable layers
			# we don't need to check for biases - checking for weights is enough
			if hasattr(self.layers[i], 'weights'):
				self.trainable_layers.append(self.layers[i])
		#}

		# update loss object with trainable layers
		self.loss.remember_trainable_layers(self.trainable_layers)

		# if output activation is Softmax and loss function is Categorical Cross-Entropy
		# create an object of combined activation and loss function containing faster gradient
		if isinstance(self.layers[-1], Activation_Softmax) and isinstance(self.loss, Loss_CategoricalCrossEntropy):
			self.softmax_classifier_output = Activation_Softmax_Loss_CategoricalCrossEntropy()


	# train the model
	def train (self, X, y, *, epochs = 1, batch_size = None, print_every = 1, validation_data = None):

		# initialize accuracy object
		self.accuracy.init(y)

		# default value if batch size is not being set
		train_steps = 1

		# if there is the validation data
		# set default number of steps for validation as well
		if validation_data is not None:

			validation_steps = 1

			# for better readability
			X_val, y_val = validation_data
		#}

		# calculate number of steps
		if batch_size is not None:

			train_steps = len(X) // batch_size

			# dividing rounds down. if there are some remaining
			# data, but not a full batch, this won't include it
			# add 1 to include this not full batch
			if train_steps * batch_size < len(X):
				train_steps += 1

			if validation_data is not None:

				validation_steps = len(X_val) // batch_size

				# dividing rounds down. if there are some remaining
				# data, but not a full batch, this won't include it
				# add 1 to include this not full batch
				if validation_steps * batch_size < len(X_val):
					validation_steps += 1
			#}
		#}

		# main training loop
		for epoch in range(1, epochs + 1):

			# print epoch number
			print(f'epoch: {epoch}')

			# reset accumulated values in loss and accuracy objects
			self.loss.new_pass()
			self.accuracy.new_pass()

			# iterate over steps
			for step in range(train_steps):

				# if batch size is not set -
				# train using one step and full dataset
				if batch_size is None:

					batch_X = X
					batch_y = y

				# otherwise slice a batch
				else:
					batch_X = X[step * batch_size : (step + 1) * batch_size]
					batch_y = y[step * batch_size : (step + 1) * batch_size]
				#}

				# perform the forward pass
				output = self.forward(batch_X, training = True)

				# calculate loss
				data_loss, regularization_loss = self.loss.calculate(output, batch_y, include_regularization = True)

				loss = data_loss + regularization_loss

				# get predictions and calculate an accuracy
				predictions = self.output_layer_activation.predictions(output)

				accuracy = self.accuracy.calculate(predictions, batch_y)

				# perform backward pass
				self.backward(output, batch_y)

				# optimize (update parameters)
				self.optimizer.pre_update_params()

				for layer in self.trainable_layers:
					self.optimizer.update_params(layer)

				self.optimizer.post_update_params()

				# print a summary
				if not step % print_every or step == train_steps - 1:
					print(f'step: {step}, acc: {accuracy:.3f}, '
						+ f'loss: {loss:.3f} (data_loss: {data_loss:.3f}, reg_loss: {regularization_loss:.3f}), '
						+ f'lr: {self.optimizer.current_learning_rate:.6f}')
			#}

			# get and print epoch loss and accuracy
			epoch_data_loss, epoch_regularization_loss = self.loss.calculate_accumulated(include_regularization = True)
			epoch_loss = epoch_data_loss + epoch_regularization_loss

			epoch_accuracy = self.accuracy.calculate_accumulated()

			print(f'training: acc: {epoch_accuracy:.3f}, '
				+ f'loss: {epoch_loss:.3f} (data_loss: {epoch_data_loss:.3f}, reg_loss: {epoch_regularization_loss:.3f}), '
				+ f'lr: {self.optimizer.current_learning_rate:.6f}')

			# if there is the validation data
			if validation_data is not None:

				# reset accumulated values in loss and accuracy objects
				self.loss.new_pass()
				self.accuracy.new_pass()

				# iterate over steps
				for step in range(validation_steps):

					# if batch size is not set -
					# validate using one step and full dataset
					if batch_size is None:

						batch_X = X_val
						batch_y = y_val

					# otherwise slice a batch
					else:
						batch_X = X_val[step * batch_size : (step + 1) * batch_size]
						batch_y = y_val[step * batch_size : (step + 1) * batch_size]
					#}

					# perform the forward pass
					output = self.forward(batch_X, training = False)

					# calculate the loss
					self.loss.calculate(output, batch_y)

					# get predictions and calculate an accuracy
					predictions = self.output_layer_activation.predictions(output)

					self.accuracy.calculate(predictions, batch_y)
				#}

				# get and print validation loss and accuracy
				validation_loss = self.loss.calculate_accumulated()
				validation_accuracy = self.accuracy.calculate_accumulated()

				print('validation: '
					+ f'acc: {validation_accuracy:.3f}, '
					+ f'loss: {validation_loss:.3f}')
			#}
		#}

	# performs forward pass
	def forward (self, X, training):

		# call forward method on the input layer
		# this will set the output property that
		# the first layer in "prev" object is expecting
		self.input_layer.forward(X, training)

		# call forward method of every object in a chain
		# pass output of the previous object as a parameter
		for layer in self.layers:
			layer.forward(layer.prev.output, training)

		# "layer" is now the last object from the list,
		# return its output
		return layer.output


	# performs backward pass
	def backward (self, output, y):

		# if softmax classifier
		if self.softmax_classifier_output is not None:

			# first call backward method on the combined activation/loss
			# this will set dinputs property
			self.softmax_classifier_output.backward(output, y)

			# since we'll not call backward method of the last layer
			# which is Softmax activation as we used combined activation/loss object
			# let's set dinputs in this object
			self.layers[-1].dinputs = self.softmax_classifier_output.dinputs

			# call backward method going through all the objects
			# in reversed order passing dinputs as a parameter
			for layer in reversed(self.layers[:-1]):
				layer.backward(layer.next.dinputs)

			return
		#}

		# first call backward method on the loss object
		# this will set dinputs property that the last
		# layer will try to access shortly
		self.loss.backward(output, y)

		# call backward method going through all the objects
		# in reversed order passing dinputs as a parameter
		for layer in reversed(self.layers):
			layer.backward(layer.next.dinputs)

