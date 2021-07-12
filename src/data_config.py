import numpy as np
import pandas as pd

from utils import _normalize_path

# To export image into output folder
APPLY_MFCC = False
EXPORT_IMAGE = False

# Source Coswara folder for metadata loading
PREFIX_INPUT = r'../data/Coswara_Data'

# Path to the extracted data
EXTRACTED_DATA_PATH = r'../data/extracted'

# MODEL PATH
PREFIX_MODEL = r'../model'
PREFIX_OUTPUT = r'../data/raw'
POSTFIX_MODEL = ''

# PREFIX_OUTPUT = r'D:\Projects\comp-7405-proj\mel_spectrum_no_mfcc'
# POSTFIX_MODEL = 'mel_norm'

# PREFIX_OUTPUT = r'D:\Projects\comp-7405-proj\mel_spectrum-mfcc-normalization'
# POSTFIX_MODEL = 'mel_mfcc_norm'

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

# RANDOM SEED
SEED = 'o-^8-$a$5yb!71f+b6s4-%&!^sy9cqjm(%n8t11=sz%=qrgwpx'

# index_col = ['covid_status', 'test_status']
index_col = ['covid_status']
key_col = ['cough-heavy']


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
            'healthy': 0,
            'no_resp_illness_exposed': 0.3,
            'positive_asymp': 0.5,
            'resp_illness_not_identified': 0.5,
            'positive_mild': 0.6,
            'positive_moderate': 0.8,
            'recovered_full': 1,
        },
        'test_status': {
            'n': 0,
            'na': 0.5,
            'p': 1,
        }
    })

    return data


def get_wav_data(data: pd.DataFrame) -> pd.DataFrame:
    for i in data.index:
        npz_raw = _normalize_path('{}/{}.npz'.format(PREFIX_OUTPUT, i))
        data.loc[i, key_col[0]] = npz_raw
    return data
