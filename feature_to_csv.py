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
import feature_extraction_util


def extract_features(dataset_dir, csv_dir):
    files = feature_extraction_util.read_file(dataset_dir)
    directory = csv_dir
    file_names = []
    dataset_gender = []
    data = []

    # Gender of the each speaker
    gender = {'19': 'F', '26': 'M', '27': 'M', '32':'F', '39': 'F', '40': 'F',
            '60': 'M', '78': 'M', '83': 'F', '87':'F', '89':'F','103':'F','118':'M',
            '125':'F','150':'F','163':'M','196':'M','198':'F','200':'F'}

    for f in files:
        y, sr = librosa.load(f)

        # Reduce the noise
        y_noise_reduced = feature_extraction_util.ambient_noise_reduction(y, sr)

        # Enhance the vocal enhancements
        y_vocal_enhanced = feature_extraction_util.vocal_enhancement(y_noise_reduced, sr)

        # Trim the audio
        y_trimmed, _ = feature_extraction_util.audio_trimming(y_vocal_enhanced, sr)

        # Get 13 MFCC values
        mfcc_features = librosa.feature.mfcc(y=y_trimmed, sr=sr,
                                            n_mfcc=13, hop_length=int(0.010 * sr),
                                            n_fft=int(0.025 * sr))

        # Take the mean of MFCC coefficients
        mfcc_mean = mfcc_features.mean(axis=1)

        # Get MFCC Delta
        mfcc_delta = librosa.feature.delta(mfcc_features, order=1)

        # Take the mean of MFCC Delta coefficients
        mfcc_delta_mean = mfcc_delta.mean(axis=1)

        # Create a list of MFCC(13 columns) and MFCC Delta coefficients(13 columns) and appned it to data list
        data.append(np.concatenate((mfcc_mean, mfcc_delta_mean)))

        # Store the name of the files to a list in order to add it to the dataframe.
        file_names.append(f.split("/")[-1].split("-")[0])

        # Store the gender of the speaker to the list in order to add it to the dataframe.
        dataset_gender.append(gender[f.split("/")[-1].split("-")[0]])

        print("the speaker %s has been added to the list!" % f.split("/")[-1].split("-")[0])

    # Pass the list of coefficients to the dataframe.
    df = pd.DataFrame(data=data)

    # Insert the column of the file name and the genders to the dataframe 
    df.insert(0, "file name", file_names, True)
    df.insert(1, "file name", dataset_gender, True)

    # Save the dataframe to the CSV file.
    df.to_csv(directory, index=False)
