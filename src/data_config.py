import os.path

import numpy as np
import pandas as pd
from tensorflow import keras

from utils import _normalize_path

# To export image into output folder
NFFT_CHUNK_SIZE = 256  # chunk size for FFT
NUM_FILTER = 40  # number of filter for Mel Spectrogram
APPLY_MFCC = True  # flag to apply MFCC
EXPORT_IMAGE = False  # to export all result to graphs

# Source Coswara folder for metadata loading
PREFIX_INPUT = r'../data/Coswara_Data'

# Path to the extracted data
EXTRACTED_DATA_PATH = r'../data/extracted'

# MODEL PATH
PREFIX_MODEL = r'../model'
#
PREFIX_OUTPUT = r'../data/raw'
POSTFIX_MODEL = '3-layers'

# PREFIX_OUTPUT = r'D:\Projects\comp-7405-proj\mel_spectrum_no_mfcc'
# POSTFIX_MODEL = 'mel_norm'

# PREFIX_OUTPUT = r'D:\Projects\comp-7405-proj\mel_spectrum-mfcc-normalization'
# POSTFIX_MODEL = '3-layers'

# PREFIX_OUTPUT = r'D:\Projects\comp-7405-proj\data_clean'
# POSTFIX_MODEL = '3-layers-clean'

BANNED_ID = [
    'XFX3DxpzWlTsqde0wmliVzvRXnf1', 'pBBuvcoBj7hjmNVYFICT4hQYRGw1', 'jaaBKWGuppchj0ahnBiFzFeuBB23',
    'Tfvspm3rapd3ZLyAoMG36VxNQnr2', 'f9g4olEAspen4dJakQJsI2EME032', 'VxQekDvwa3N5B2FqB4FlQ5jBjTp2',
    'I5gTlFVoKbVFGCKxwV6QXSOdkZN2', 'mNcNjQMsv8aZXFGuguWbLdmkOQk2', 'FJTHLyF7sGYFJpundI1mZ28HQQ03',
    '8Suz7Jl7GtXpdB7UeIPtLuZb1Lp2', 'C3luMlCgAFZqGmGjFscXe9fVHSG3', 'TRK6gUseEUS7dbF0soG80W1uLm53',
    '9XKj7fAmvwPUas9GFPZuTpev7T03', 'C7Km0KttQRMMM6UoyocajfgZAOB3', 'C7Km0KttQRMMM6UoyocajfgZAOB3',
    '697xpWya0DbSEN4Y6tsEH3BchHw1', '9Rhe6RDHuMNXaGiqKVaspw71Exo1', 'cbmcTV6z0NVgO0i6yeYyAO7AoI72',
    'sbdVe2aEGKeFcPXdhSf5QOuA5qA3', 'SG95RAgm0wY3bzyIZPPSpHwyYuD3', 'TLfuOcZh0HfnVRRdez4CxLFiPki2',
    'Vq1h51z5x3Wp4wS2pCm6yEAZvu82', '8Iug9hOgiBZTCKZuDxazXlamW0g1', 'zFDUIa5rb2OG7IXXSI5vXVsoLa32',
    'DycfRd7vyobY2PgwT3rQOemSbP53', 'JUVJnOVJKKcdzppK3hkci4TO4ij1', 'kgjTguvo3vZJTO7F1qO9GxEicbA3',
    'Ts5Rbl9h9pWKqCQPJwoTduGvjMm2', 'c18b81Qa5YY2RbEzblDNxMNQE312', '2XgDqzQkqLX9SbmSHfbqdRxSmRD2',
    'abpYyjmAQAMGlnyqXPGIL9lbnQo2', 'Qriv4y0rwfWRZDatFMOj9zdXeB43', 'j1vkslz2yvP8MMGGhjPnZE4CVlg2',
    'imhxF3UQDZNVEnNeyw8jOAsgtjv2', 'HdJdEWQecehLzEQcyDV7vjr85C82', 'V3EIT06H4JN5KwoK8aGRXQNzGRi1',
    'CFwFsoyLtGUxPNPw5vMD7cznAnO2', 'QlMu7Fl8iIgOZM3QHEOzOJY6npH3', 'htQzROl26OWQpIYFDzv11F79PLR2',
    'htQzROl26OWQpIYFDzv11F79PLR2', 'UiUUhL0PMjWVA6W2dKte1DCE6wG2', 'tiKv850hJFTmpCAvfJOmOOqkeUs1',
    'zvXkuEaPb0OEgG4EHx59NqdmamR2', 'U98J2q0NnycYzzUS2BYZpUa55X83',
]

BANNED_ID_BY_FEAT = {
    'breathing-deep': [],
    'breathing-shallow': [
        'W9x1xBd8ZCggMQLy9vNSgo8zeAI3', '6T43bddKoKfG7MwnJWvrPZSsyrc2', 'aU8pLZV1OUQJV0GKeeIlgWhYWeA2',
        'yZuoG6z3pRfycT6JKqwqKDMj4tM2', 'yiVfb7qpTOXWhTQITq6P4zqGtXt2'
    ],
    'cough-heavy': [],
    'cough-shallow': [
        'jSb7SyucSmTHhzs3qQoBExRMQZ02', '9hftEYixyhP1Neeq3fB7ZwITQC53', 'aU8pLZV1OUQJV0GKeeIlgWhYWeA2',
        'CdU4pgCdFcZxtDHTpLUn1mO9J3o2', '9z2XQAVyIkb0saZVigWBr3MsDcr1'
    ]}

# RANDOM SEED
SEED = 'o-^8-$a$5yb!71f+b6s4-%&!^sy9cqjm(%n8t11=sz%=qrgwpx'

# key columns for testings
index_col = ['covid_status']
key_col = [
    'breathing-deep',
    'breathing-shallow',
    'cough-heavy',
    'cough-shallow',
]


####################################################
#   Input data
####################################################

def get_data():
    # loading data
    df = pd.read_csv(_normalize_path('{}/combined_data.csv'.format(PREFIX_INPUT)))

    # dropping unrelated fields
    df.drop(
        [
            'a',  # Age (number)
            'g',  # Gender (male/female/other)
            'ep',  # Proficient in English (y/n)
            'l_c',  # Country
            'l_l',  # Locality
            'l_s',  # State
            'rU',  # Returning User (y/n)
            'um',  # Using Mask (y/n)

        ],
        axis=1,
        inplace=True)

    # setup output columns and training columns
    df = df.loc[~df['id'].isin(BANNED_ID)]
    df = df.set_index('id')
    data = df.loc[:, index_col]
    data = data.replace(np.nan, 'na')
    # assign some value for each status
    data = data.replace({
        'covid_status': {
            'healthy': 1,
            'no_resp_illness_exposed': 1,
            'resp_illness_not_identified': 1,
            'recovered_full': 2,
            'positive_asymp': 3,
            'positive_mild': 3,
            'positive_moderate': 3,
        }
    })
    return data


def get_wav_data(data: pd.DataFrame, feat: str) -> pd.DataFrame:
    for i in data.index:
        npz_raw = _normalize_path('{prefix}/{feature}/{id}.npz'.format(prefix=PREFIX_OUTPUT, feature=feat, id=i))
        if os.path.exists(npz_raw):
            data.loc[i, feat] = npz_raw
        else:
            data.loc[i, feat] = pd.NaT
    return data.dropna()


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, X_list, y_list, batch_size=32, dim=(32, 32, 32), n_channels=1, n_classes=10):
        'Initialization'
        self.X_list = X_list
        self.y_list = y_list
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.X_list) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_files = [self.X_list[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_files, indexes)

        return X, y

    def __data_generation(self, list_files, indexes):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=float)

        # Generate data
        i = 0
        for f in enumerate(list_files):
            # Store sample
            file = np.load(f[1][0])['arr_0']
            X[i,] = file.reshape(file.shape[0], file.shape[1], 1)
            i += 1

        for i in range(len(indexes)):
            # Store class
            y[i] = self.y_list[indexes[i]]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.X_list))
