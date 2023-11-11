# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 16:37:43 2023

@author: 1618047
"""

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from keras.applications import VGG16
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

import os
import matplotlib.pyplot as plt

train = 'train'
val =   'val'
test =  'test'
task4 = 'task4'

# The shape of the RGB image
img_width, img_height, channels = 64, 64, 3

# input shape
input_shape = (img_width, img_height, 3)
# position matters!
# Number_of_channels can be at the first or the last position
# in our case - "channels last"

# minibatch size
batch_size = 64
# train set size
nb_train_samples = len(os.listdir(train+'/dogs'))
# validation set size
nb_validation_samples = len(os.listdir(val+'/dogs'))
# test set size
nb_test_samples = len(os.listdir(test+'/dogs'))

nb_task4_samples = len(os.listdir(test+'/dogs'))

datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = datagen.flow_from_directory(
    train,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

val_generator = datagen.flow_from_directory(
    val,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

test_generator = datagen.flow_from_directory(
    test,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

task4_generator = datagen.flow_from_directory(
    task4,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

model = Sequential(
    [
        keras.Input(shape=input_shape),
        Conv2D(16, kernel_size=(3, 3), activation="relu"),              # 1) A convolutional layer with 16 neurons, filter size 3x3. Activation function - 'relu'
        MaxPooling2D(pool_size=(2, 2)),                                 # 2) MaxPooling layer with filter size 2x2
        Conv2D(32, kernel_size=(3, 3), activation="relu"),              # 3) A convolutional layer with 32 neurons, filter size 3x3. Activation function - 'relu'
        MaxPooling2D(pool_size=(2, 2)),                                 # 4) MaxPooling layer with filter size 2x2
        Conv2D(64, kernel_size=(3, 3), activation="relu"),              # 5) A convolutional layer with 64 neurons, filter size 3x3. Activation function - 'relu'
        MaxPooling2D(pool_size=(2, 2)),                                 # 6) MaxPooling layer with filter size 2x2
        Flatten(),                                                      # 7) Operation model.add (Flatten ()), which makes a one-dimensional vector of the resulting feature maps
        Dense(64, activation="relu"),                                   # 8) A fully connected layer with 64 neurons. Activation function - 'relu'
        Dropout(0.5),                                                   # 9) Use model.add (Dropout (0.5)) which excludes the edge from the current layer in the computational graph with a 50% probability to avoid overfitting
        Dense(1, activation="sigmoid"),
    ]
)

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=15, # try different number of epochs: 10, 15, 20; check the loss and accuracy;
    validation_data=val_generator,
    validation_steps=nb_validation_samples // batch_size)

scores = model.evaluate_generator(task4_generator, nb_test_samples // batch_size)
print("Accuracy on test data: %.2f%%" % (scores[1]*100))

plt.figure(figsize=(10, 7), dpi=300)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.figure(figsize=(10, 7), dpi=300)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()