# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 13:35:01 2020

@author: KaiShengChoi
"""

import tensorflow as tf
from tensorflow.keras import models, layers, regularizers
import audio
from mfcc import mfcc
import numpy as np
import constant


def predict(model_name, filename):
    
    samples, sample_rate = audio.decode_wav(filename)
    coefficients = mfcc(samples, sample_rate)
    coefficients = tf.reshape(tf.cast(coefficients, tf.float32),[1,98,13,1])
    model = tf.keras.models.load_model(model_name)
    prediction = model.predict(coefficients)
    
    if np.argmax(prediction) >= 0.8:
       output = constant.train_commands[np.argmax(prediction)]
    else:
       output = None
        
    return output

if __name__ == "__main__":
    audio.record("test.wav")
    prediction = predict('./models/model2.h5', 'test.wav')
    print(prediction)
    