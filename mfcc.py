# -*- coding: utf-8 -*-
"""
This file implement Mel-Frequency Cepstrum Coefficients (MFCC) using APIs from 
Tensorflow. Using APIs from Tensorflow allow MFCC calculation on GPU.

Created on Thu Jan 30 19:41:31 2020

@author: KaiShengChoi
"""

import tensorflow as tf 
import constant
import audio

def mfcc (samples, sample_rate):   
    """
    
    Descriptions
    ------------
    Calculate MFCC of sample voice
    
    
    Parameters
    ----------
    samples : tensor 
        raw data of the voice for which MFCC to be calculated. 
    sample_rate : integer
        sample rate of input voice samples

    Returns
    -------
    mfccs : tensor 
        mfcc of voice.

    """

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
    
    return mfccs


if __name__ == "__main__":
    filename = constant.dataset_path + "/bed/0a7c2a8d_nohash_0.wav"
    samples, sample_rate = audio.decode_wav(filename)
    coeff = mfcc(samples, sample_rate)
    
