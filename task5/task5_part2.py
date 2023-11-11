# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 17:48:38 2023

@author: 1618047
"""

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from keras.applications import VGG16, MobileNetV2
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

base_model = VGG16(weights='imagenet',
                  include_top=False,      
                  input_shape=(64, 64, 3))
base_model.trainable = False 

# add layers to VGG16:
inputs = keras.Input(shape=(64, 64, 3))
x = base_model(inputs, training=False)
x = Flatten()(x)                            # + flattening
x = Dense(256, activation="relu")(x)        # + Dense fullyconnected layer with 256 neurons + ReLu
x = Dropout(0.5)(x)                         # + Dropout
outputs = Dense(1, activation="sigmoid")(x) # + Dense layer with 1 neuron + sigmoid
model = keras.Model(inputs, outputs)

model.compile(loss='binary_crossentropy',
              optimizer=Adam(lr=1e-5),
              metrics=['accuracy'])

history_VGG16 = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=10,
    validation_data=val_generator,
    validation_steps=nb_validation_samples // batch_size)

scores = model.evaluate_generator(test_generator, nb_test_samples // batch_size)
print("Accuracy on test data: %.2f%%" % (scores[1]*100))

plt.figure(figsize=(10, 7), dpi=300)
plt.plot(history_VGG16.history['accuracy'])
plt.plot(history_VGG16.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.figure(figsize=(10, 7), dpi=300)
plt.plot(history_VGG16.history['loss'])
plt.plot(history_VGG16.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()