import numpy as np
import keras
from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical

# Needs to be changed to accept the input
#(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
training_data = np.load('training.npy')
training_labels = np.load('training_labels.npy')

network = models.Sequential()
network.add(layers.Dense(10, activation='relu', input_shape=(1 * 10,)))
network.add(layers.Dense(320, activation='relu', input_shape=(10 * 32,)))
network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

for i in range(len(training_data)):
    network.fit(training_data[i], training_labels, epochs=5, batch_size=128)
test_loss, test_acc = network.evaluate(training_data[0], training_labels)
print('test_acc:', test_acc, 'test_loss', test_loss)

# network = models.Sequential()
# network.add(layers.Dense(784, activation='relu', input_shape=(28 * 28,)))
# network.add(layers.Dense(784, activation='relu', input_shape=(28 * 28,)))
# network.add(layers.Dense(10, activation='softmax'))
# network.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# train_images = train_images.reshape((60000, 28 * 28))
# train_images = train_images.astype('float32')
# test_images = test_images.reshape((10000, 28 * 28))
# test_images = test_images.astype('float32')
# train_labels = to_categorical(train_labels)
# test_labels = to_categorical(test_labels)
# network.fit(train_images, train_labels, epochs=5, batch_size=128)
# test_loss, test_acc = network.evaluate(test_images, test_labels)
# print('test_acc:', test_acc, 'test_loss', test_loss)

# def get_model():
#     inputs = keras.Input(shape=(32,))
#     outputs = keras.layers.Dense(1)(inputs)
#     model = keras.Model(inputs, outputs)
#     model.compile(optimizer="adam", loss="mean_squared_error")
#     return model

# model = get_model()

# test_input = np.random.random((128, 32))
# test_target = np.random.random((128, 1))
# model.fit(test_input, test_target, epochs=5, batch_size=32)
