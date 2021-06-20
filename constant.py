# -*- coding: utf-8 -*-
"""
The file consists of configurations and constants for Keyword Sporting System.

Created on Fri Jan 31 14:15:20 2020

@author: KaiShengChoi
"""

import pyaudio

"""
The path of Google Speech Command Datasets
"""

dataset_path = "/home/derrick/Code/speech_commands_v0.01"


"""
List of commands of Google Commands Datasets v0.01
"""

commands = ["yes", "no", "up", "down", "left","right", "on", "off", "stop", 
            "go", "zero", "one", "two", "three", "four", "five", "six", 
            "seven", "eight", "nine", "bed", "bird", "cat", "dog", "happy", 
            "house","marvin", "sheila","tree", "wow"]

"""
List of commands to be trained on
"""

train_commands = ["zero", "one", "two", "three", "four", "five", "six", 
                  "seven","eight","nine"]

"""
WAV configuration
"""

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 160
RECORD_SECONDS = 1

"""
MFCC Parameters
"""

#desired number of samples per voices
#the voice will be padded with zeros at the end if the number of sample less then this
desired_samples = 16000

#configuration for short time fourier transform
frame_length = 400         #25ms
frame_step = 160           #10ms
fft_length = frame_length  #25ms

#lower edge frequency of spectrum
lower_edge_hertz = 40.0

#upper edge frequency of spectrum
upper_edge_hertz = 4000.0

#number of bands in mel-scale
num_mel_bins = 80

#number of output of MFCC
num_mfccs = 13

"""
Model's Architecture, Parameters and Hyperparameter
"""



