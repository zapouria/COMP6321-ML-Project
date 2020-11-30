import librosa
import numpy as np
import math
from pysndfx import AudioEffectsChain
import python_speech_features
import matplotlib.pyplot as plt 
import librosa.display
import pandas as pd
import IPython.display as ipd
import warnings
import sklearn
from sklearn.preprocessing import MinMaxScaler


def read_file(loc):
    """
    This function read the audio files and the outputs are sr=22050:22kHz, the sampling frequency, and x is the signal
    """
    files = librosa.util.find_files(loc, ext=['wav']) 
    files = np.asarray(files)
    return files


def ambient_noise_reduction(y, sr):
    """
    This function is based on the method presented on Ambient Noise Reduction in the Master thesis reference.
    The goal is to reduce the noise which exists in the background of the environment.
    First finds the centroids of the frames, then their minimum and maximum to use them
    as the upper and lower thresholds respectively.
    At the end, it decreases the gains of the noisy part of the audio.
    """
    sc = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=int(0.025 * sr),
                                           hop_length=int(0.010 * sr))

    # This code finds the centroids of each frame with the length of 25ms and hop length of 10ms
    upper_threshold = np.max(sc)
    lower_threshold = np.min(sc)
    noise_reduced = AudioEffectsChain().lowshelf(gain=-30.0, frequency=lower_threshold).highshelf(gain=-30.0, frequency=upper_threshold).limiter(gain=10.0)
    return noise_reduced(y)


def vocal_enhancement(y, sr):
    """
    In this function we want to enhace the significant parts of the audio that we wish to use
    to extract features from, and build machine learning classifiers. It will be done using concepts of MFCCs again.
    Finding the strongest frame based on the sum square of the MFCCs coefficients, then add a gain to it.
    """
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_fft=int(0.025 * sr),
                                 hop_length=int(0.010 * sr))
    sum_of_squares = [0] * len(mfccs)
    for idx, val in enumerate(mfccs):
        for d in val:
            sum_of_squares[idx] = sum_of_squares[idx] + d**2

    strongest_frame = sum_of_squares.index(max(sum_of_squares))
    hz = python_speech_features.base.mel2hz(mfccs[strongest_frame])
    minimum_hz = min(hz)
    vocal_enhanced = AudioEffectsChain().lowshelf(frequency=minimum_hz*(-1), gain=10.0)
    return vocal_enhanced(y)


def audio_trimming(y, sr):
    """
    This function is desiged to eliminate the silence parts of the audio, and is based on the master thesis reference.
    we set the threshold value of 20 decibels, the frame length as 2048 and the hop length as 500.
    """
    return librosa.effects.trim(y=y, top_db=20, frame_length=2048, hop_length=500)
