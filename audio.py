# -*- coding: utf-8 -*-
"""
This file contains methods to decode wav files and record voice to wav file.
The Wav file decoding uses the API from Tensorflow whereas recording uses the 
API from Pyadio and Wave.

Created on Tue Feb  4 13:19:58 2020

@author: KaiShengChoi
"""

import pyaudio
import wave
import constant
import tensorflow as tf

def decode_wav (filename):
    """
    
    Descriptions
    ------------
    decode and return the raw voice samples from WAV file.
    
    
    Parameters
    ----------
    filename : string 
        path to WAV file 

    Returns
    -------
    samples : tensor 
        raw data of the voice for which MFCC to be calculated. 
    sample_rate : integer
        sample rate of input voice samples

    """
    raw_audio = tf.io.read_file(filename)
    samples, sample_rate = tf.audio.decode_wav(contents = raw_audio, \
        desired_samples = constant.desired_samples)
    return samples, sample_rate
 

def record(filename):
    """
    
    Descriptions
    ------------
    record voice to WAV file
    

    Parameters
    ----------
    filename : string
        path to the WAV file in which recorded voice saved.

    Returns
    -------
    NONE

    """
    audio = pyaudio.PyAudio()
 
    # start Recording
    stream = audio.open(format = constant.FORMAT, channels = constant.CHANNELS,
                rate=constant.RATE, input=True,
                frames_per_buffer = constant.CHUNK)
  
    frames = []
    
    print("Start Recording")
 
    for i in range(0, int(constant.RATE / constant.CHUNK * constant.RECORD_SECONDS)):
        data = stream.read(constant.CHUNK)
        frames.append(data)

    print("Stop Recording")
    # stop Recording
    stream.stop_stream()
    stream.close()
    audio.terminate()
 
    waveFile = wave.open(filename, 'wb')
    waveFile.setnchannels(constant.CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(constant.FORMAT))
    waveFile.setframerate(constant.RATE)
    waveFile.setnframes(constant.RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()
    
if __name__ == "__main__":
    record("test.wav")
    