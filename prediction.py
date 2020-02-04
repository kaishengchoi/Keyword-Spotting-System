# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 13:35:01 2020

@author: KaiShengChoi
"""

import tensorflow as tf
import audio
from mfcc import mfcc
import numpy as np
import constant


def predict(model_name, filename):
    
    audio.record(filename)
    coefficients = mfcc(filename)
    coefficients = tf.reshape(tf.cast(coefficients, tf.float32),[1,98,13,1])
    model = tf.keras.models.load_model(model_name)
    prediction = model.predict(coefficients)
    
    if np.argmax(prediction) >= 0.8:
        output = constant.train_commands[np.argmax(prediction)]
    else:
        output = None
        
    return output

if __name__ == "__main__":
    prediction = predict('model2.h5', 'test.wav')
    print(prediction)
    