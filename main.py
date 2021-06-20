# -*- coding: utf-8 -*-
"""

Created on Mon Jan 27 19:57:06 2020

@author: KaiShengChoi
"""

import tensorflow as tf
import constant
import audio as au
import mfcc
import dataset
#import train
import prediction
import pyaudio
import webrtcvad
import wave
import numpy as np

physical_devices = tf.config.experimental.list_physical_devices('GPU') 
for physical_device in physical_devices: 
    tf.config.experimental.set_memory_growth(physical_device, True)

vad = webrtcvad.Vad()
vad.set_mode(1)

audio = pyaudio.PyAudio()
 
# start Recording
stream = audio.open(format = constant.FORMAT, channels = constant.CHANNELS,
        rate=constant.RATE, input=True,
        frames_per_buffer = constant.CHUNK)

frames = []

model = tf.keras.models.load_model('./models/model2.h5')
  
print("Start Recording")

while True:
    frames.append(stream.read(constant.CHUNK))
    test = False
    print(frames[-1])
    test = vad.is_speech(frames[-1], constant.RATE)
    # print("Contains speech: %s"%(test))
    # if test == True:
    #     for i in range(0, int((constant.RATE / constant.CHUNK * constant.RECORD_SECONDS))):
    #         frames.append(stream.read(constant.CHUNK))
          
    #     waveFile = wave.open("test.wav", 'wb')
    #     waveFile.setnchannels(constant.CHANNELS)
    #     waveFile.setsampwidth(audio.get_sample_size(constant.FORMAT))
    #     waveFile.setframerate(constant.RATE)
    #     waveFile.setnframes(constant.RATE)
    #     waveFile.writeframes(b''.join(frames))
    #     waveFile.close()
        
    #     samples, sample_rate = au.decode_wav("test.wav")
    #     coefficients = mfcc.mfcc(samples, sample_rate)
    #     coefficients = tf.reshape(tf.cast(coefficients, tf.float32),[1,98,13,1])
    #     pre = model.predict(coefficients)
        
    #     #if np.argmax(pre) >= 0.95:
    #     output = constant.train_commands[np.argmax(pre)]
    #     #else:
    #     #    output = None
            
    #     print(output)
    # while len(frames) > 4: frames.pop(0)
        
        
# vad = webrtcvad.Vad()
# vad.set_mode(3)
# # Run the VAD on 10 ms of silence. The result should be False.
# sample_rate = 16000
# frame_duration = 10  # ms
# frame = b'\x00\x00' * int(sample_rate * frame_duration / 1000)
# print(len(frame))
# print ('Contains speech: %s' % (vad.is_speech(frame, sample_rate)))