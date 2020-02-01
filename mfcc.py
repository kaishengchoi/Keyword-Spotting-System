# -*- coding: utf-8 -*-
"""
This file implement Mel-Frequency Cepstrum Coefficients (MFCC) using APIs from 
Tensorflow. Using APIs from Tensorflow allow MFCC calculation on GPU.

Created on Thu Jan 30 19:41:31 2020

@author: KaiShengChoi
"""

import os
import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt
import constant


def mfcc (filename):   
    """
    
    Descriptions
    ------------
    Calculate MFCC of WAV voice file
    
    
    Parameters
    ----------
    filename : string
        path of the wav voice file.

    Returns
    -------
    mfccs.numpy()[0] :numpy of float32
        mfcc of voice.

    """
    raw_audio = tf.io.read_file(filename)
    samples, sample_rate = tf.audio.decode_wav(contents = raw_audio, \
        desired_samples = constant.desired_samples)
    #transpose the dimension of samples
    signals = tf.reshape(samples, [1, -1])
    
    # STFT
    spectrogram = tf.signal.stft(signals, frame_length = constant.frame_length, \
        frame_step = constant.frame_step, fft_length = constant.fft_length)
    magnitude_spectrograms = tf.abs(spectrogram)
    
    # Warp the linear scale spectrograms into the mel-scale.
    num_spectrogram_bins = magnitude_spectrograms.shape[-1]
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix( \
        constant.num_mel_bins, num_spectrogram_bins, sample_rate, \
        constant.lower_edge_hertz, constant.upper_edge_hertz)
        
    mel_spectrograms = tf.tensordot(magnitude_spectrograms, \
        linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(magnitude_spectrograms.shape[:-1].concatenate( \
        linear_to_mel_weight_matrix.shape[-1:]))
    
    # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
    
    # Compute MFCCs from log_mel_spectrograms and take the first 13.
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms( \
        log_mel_spectrograms)[..., :constant.num_mfccs]
    
    return mfccs.numpy()[0]


if __name__ == "__main__":
    keyword = "bed"
    folders = os.listdir("G:\\Keyword Spotting System\\speech_commands_v0.01" + "\\" + keyword)
    for files in folders:
        filename = str(constant.dataset_path) + '\\' + keyword +'\\' + files
        mfcc_bed = mfcc(filename)
        fig = plt.figure(figsize=(14,14))
        plt.ylabel("MFCC (log) coefficient")
        plt.imshow(np.swapaxes(mfcc_bed,0,1))