import numpy as np
import pandas as pd

from utils import _normalize_path

PREFIX_INPUT = r'../data/Coswara_Data'
PREFIX_OUTPUT = r'../data/raw'
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
    'Ts5Rbl9h9pWKqCQPJwoTduGvjMm2', 'c18b81Qa5YY2RbEzblDNxMNQE312', ]

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
    for id in data.index:
        npz_raw = _normalize_path('{}/{}.npz'.format(PREFIX_OUTPUT, id))
        data.loc[id, key_col[0]] = npz_raw
    return data
