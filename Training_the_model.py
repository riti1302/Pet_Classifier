#!/usr/bin/env python
# coding: utf-8


import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
from tensorflow.keras.callbacks import TensorBoard
import time

NAME = "Cats-vs-Dogs-cnn-64x2-{}".format(int(time.time()))

X  = pickle.load(open("X.pickle", "rb"))
y  = pickle.load(open("y.pickle", "rb"))


X = X/225.0

dense_layers = [0]
layer_sizes = [128]
conv_layers = [1]

for dense_layer in dense_layers:
	for layer_size in layer_sizes:
		for conv_layer in conv_layers:
			NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
			print(NAME)

			tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))
			model = Sequential()

			model.add(Conv2D(layer_size, (3,3), input_shape = X.shape[1:]))
			model.add(Activation("relu"))
			model.add(MaxPooling2D(pool_size = (2,2)))

			for l in range(conv_layer-1):
				model.add(Conv2D(layer_size, (3,3)))
				model.add(Activation("relu"))
				model.add(MaxPooling2D(pool_size = (2,2)))

			model.add(Flatten())
			for l in range(dense_layer):
				model.add(Dense(layer_size))
				model.add(Activation("relu"))

			model.add(Dense(1))
			model.add(Activation("sigmoid"))

			model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ['accuracy'])

			model.fit(X, y, batch_size = 32, epochs = 10, validation_split = 0.3, callbacks = [tensorboard])




