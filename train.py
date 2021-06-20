# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 18:06:28 2020

@author: derri
"""
#%%
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models, layers, regularizers
from dataset import dataset_sort, dataset_gen, train_list, get_dataset

physical_devices = tf.config.experimental.list_physical_devices('GPU') 
for physical_device in physical_devices: 
    tf.config.experimental.set_memory_growth(physical_device, True)
    
tf.keras.backend.clear_session() 


train_dataset,train_label, testing_dataset, testing_label ,validation_dataset, validation_label = get_dataset()


#%%
model1 = models.Sequential()
model1.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape = (98,13,1)))
model1.add(layers.MaxPooling2D((2, 2)))
model1.add(layers.Conv2D(32, (3, 3), activation='relu'))
model1.add(layers.MaxPooling2D((2, 2)))
model1.add(layers.Flatten())
model1.add(layers.Dense(64, activation='relu'))
model1.add(layers.Dense(10, activation='softmax'))
model1.summary()

model1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


history1 = model1.fit(train_dataset , train_label, epochs = 40, 
                    validation_data = (validation_dataset, validation_label))
#%%
model2 = models.Sequential()
model2.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape = (98,13,1), kernel_regularizer=regularizers.l2(0.001)))
model2.add(layers.MaxPooling2D((2, 2)))
model2.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model2.add(layers.MaxPooling2D((2, 2)))
model2.add(layers.Flatten())
model2.add(layers.Dropout(0.2))
model2.add(layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model2.add(layers.Dropout(0.2))
model2.add(layers.Dense(10, activation='softmax', kernel_regularizer=regularizers.l2(0.001)))
model2.summary()

model2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history2 = model2.fit(train_dataset , train_label, epochs = 80, 
                    validation_data = (validation_dataset, validation_label))
#%%
plt.plot(history1.history['accuracy'], label='model1_accuracy')
plt.plot(history1.history['val_accuracy'], label = 'model1_val_accuracy')
plt.plot(history2.history['accuracy'], label='model2_accuracy')
plt.plot(history2.history['val_accuracy'], label = 'model2_val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

results1 = model1.evaluate(testing_dataset, testing_label, verbose = 0)
print("model1 ", results1)
results2 = model2.evaluate(testing_dataset, testing_label, verbose = 0)
print("model2 ", results2)

model1.save('./models/model1.h5') 
model2.save('./models/model2.h5') 