import os
import random
from datetime import datetime
from typing import Tuple, List

import numpy as np
from scipy import signal
from scipy.fft import dct


def _normalize_path(filepath: str) -> str:
    result = os.path.expanduser(filepath) if '~' in filepath else filepath
    return os.path.realpath(os.path.abspath(result))


def stft_spectrogram(rate, audio, NFFT=256) -> Tuple[np.ndarray, ...]:
    f, t, Zxx = signal.stft(audio, fs=rate, nperseg=NFFT)
    Zxx = np.abs(Zxx)  # FTT Magnitude = |Zxx|
    Zxx = ((1.0 / NFFT) * ((Zxx) ** 2))  # Power Spectrum = |ZXX| ^ 2 / N
    return t, f, Zxx


def mel_filter(Zxx: np.ndarray, sample_rate: int, NFFT: int = 256, nfilt: int = 40) -> np.ndarray:
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10 ** (mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = np.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])  # left
        f_m = int(bin[m])  # center
        f_m_plus = int(bin[m + 1])  # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = np.dot(fbank, Zxx)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * np.log10(filter_banks)  # dB
    filter_banks *= (filter_banks > 0)
    return filter_banks


def mfcc(filter_banks: np.ndarray, num_ceps=20):
    # Mel-frequency Cepstral Coefficients (MFCCs)
    mfcc = dct(filter_banks, type=2, axis=0, norm='ortho')
    mfcc *= (mfcc > 0)
    mfcc = mfcc[1: (num_ceps + 1), :]  # Keep 2-20
    return mfcc


def prepare_data(data: np.ndarray,
                 ratio: float,
                 index_col: List[str],
                 key_col: str,
                 randomize: bool = False,
                 seed=datetime.now()) -> Tuple[np.ndarray, ...]:
    # split the data into training set and test set
    # use 75 percent of the data to train the model and hold back 25 percent for testing
    train_ratio = ratio

    # number of samples in the data_subset
    num_rows = data.shape[0]
    train_set_size = int(num_rows * train_ratio)
    indices = list(range(num_rows))

    # If randomize is true, then we generate re-arrange the list randomly
    if randomize:
        # shuffle the indices
        random.seed(seed)
        random.shuffle(indices)

    train_indices = indices[:train_set_size]
    test_indices = indices[train_set_size:]

    # create training set and test set
    train_data = data.iloc[train_indices, :]
    test_data = data.iloc[test_indices, :]
    print('{} training samples + {} test samples'.format(train_data.shape, test_data.shape))

    # If key columns is provided, select only columns in key_col set,
    # otherwise select everything that's not index columns
    X_test_data = test_data.loc[:, key_col].to_numpy()
    X_train_data = train_data.loc[:, key_col].to_numpy()

    y_test_data = test_data.loc[:, index_col].to_numpy()
    y_train_data = train_data.loc[:, index_col].to_numpy()

    return (X_test_data, y_test_data, X_train_data, y_train_data)
