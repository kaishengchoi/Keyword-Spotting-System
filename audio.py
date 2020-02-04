# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 13:19:58 2020

@author: KaiShengChoi
"""

import pyaudio
import wave
import constant
 

def record(filename):
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
    record("G:\\Keyword Spotting System\\code\\test.wav")