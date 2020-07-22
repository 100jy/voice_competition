import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from glob import glob
import librosa


def make_data(datalist, feature='signal'):
    arr = []
    # signal data
    if feature == 'signal':
        for file in tqdm(datalist):
            # 16gb ram 기준
            # float32, sr = 22050 -> OOM
            # float32, sr = 16000 -> OOM
            # float32, sr = 11025 -> success
            x, sr = librosa.load(file, sr=11025)
            arr.append(x)

    # spectogam data
    elif feature == 'spectogram':
        for file in tqdm(datalist):
            x, sr = librosa.load(file, sr=11025)
            spec = librosa.stft(x)
            arr.append(spec.flatten())

    # MFCC data
    elif feature == 'MFCC':
        hop_size = int(22050/79)
        for file in tqdm(datalist):
            x, sr = librosa.load(file)
            mfcc = librosa.feature.mfcc(x, sr=sr, n_mfcc=80, fmax=3000,hop_length = hop_size)
            arr.append(mfcc.flatten())

    else:
        raise ValueError

    result = np.array(arr)

    return result

if __name__ == '__main__':
    raw_data = glob('./train/*.wav')
    mfcc = make_data(raw_data,feature='MFCC')
    pd.DataFrame(mfcc).to_pickle('./data/x_trian_mfcc_80.pickle')
    
    raw_tet = glob('./test/*.wav')
    mfcc = make_data(raw_tet,feature='MFCC')
    pd.DataFrame(mfcc).to_pickle('./data/x_test_mfcc_80.pickle')
    
