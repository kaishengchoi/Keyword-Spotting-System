# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 18:06:28 2020

@author: derri
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models, layers
from dataset import dataset_sort, dataset_gen, train_list

train_list()
dataset_sort()
train_dataset,train_label, testing_dataset, testing_label ,validation_dataset, validation_label = dataset_gen()

train_dataset = tf.reshape(train_dataset, [18620, 98, 13, 1])
testing_dataset = tf.reshape(testing_dataset, [2552, 98, 13, 1])
validation_dataset = tf.reshape(validation_dataset,[2494, 98, 13,1])

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape = (98,13,1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.summary()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_dataset , train_label, epochs = 100, 
                    validation_data = (validation_dataset, validation_label))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

