import os

import numpy as np
from matplotlib import pyplot as plt
from scipy.io import wavfile

from data_config import PREFIX_OUTPUT, BANNED_ID, BANNED_ID_BY_FEAT, \
    key_col, \
    EXPORT_IMAGE, \
    EXTRACTED_DATA_PATH, \
    APPLY_MFCC, \
    NFFT_CHUNK_SIZE, \
    NUM_FILTER
from utils import _normalize_path, stft_spectrogram, mel_filter, mfcc


def spec(t: np.ndarray, f: np.ndarray, Zxx: np.ndarray, output: str) -> None:
    plt.clf()
    plt.pcolormesh(t, f, Zxx, vmin=0, vmax=2 * np.sqrt(2), shading='auto')
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    if output:
        plt.savefig(output)
    else:
        plt.show()


def mel_spec(t: np.ndarray, f: np.ndarray, Zxx: np.ndarray, output: str) -> None:
    plt.clf()
    plt.pcolormesh(t, f, Zxx, vmin=0, vmax=2 * np.sqrt(2), shading='auto')
    plt.title('Mel Log Spectrum')
    plt.ylabel('Mel Filter Bands')
    plt.xlabel('Time [sec]')
    if output:
        plt.savefig(output)
    else:
        plt.show()


def mfcc_spec(t: np.ndarray, f: np.ndarray, Zxx: np.ndarray, output: str) -> None:
    plt.clf()
    plt.pcolormesh(t, f, Zxx, vmin=0, vmax=2 * np.sqrt(2), shading='gouraud')
    plt.title('Mel-frequency Cepstral Coefficients')
    plt.ylabel('MFCC Coefficients')
    plt.xlabel('Time [sec]')
    if output:
        plt.savefig(output)
    else:
        plt.show()


def main():
    ban = []
    ban_by_feat = {}
    index_wav = {}

    # scrambling through the extracted WAV files and map them with corresponded patient ID
    for path, folder, files in os.walk(_normalize_path(EXTRACTED_DATA_PATH)):
        # extract sample ID from path
        id = os.path.basename(path)
        if id in BANNED_ID:
            continue

        # starting process each wav files in path
        wav_files = [f for f in files if f.endswith('.wav') and not f.startswith('._')]
        for f in wav_files:
            # extract feature name - wav file name
            feature = f.split('.')[0]

            # skip feature if not defined in key_col
            if key_col and feature not in key_col:
                continue

            if feature in BANNED_ID_BY_FEAT and id in BANNED_ID_BY_FEAT[feature]:
                continue

            #
            wav_path = _normalize_path('{}/{}'.format(path, f))
            print(wav_path)

            ####################################################
            #   Sound processing
            ####################################################
            try:
                # read wav file and parse into spectrogram
                # Fourier-Transform and Power Spectrum
                sample_rate, audio = wavfile.read(wav_path)
                t, f, spectrogram = stft_spectrogram(sample_rate, audio, NFFT=NFFT_CHUNK_SIZE)

                # Mel Filter Banks power spectrum
                mel_spectrum = mel_filter(spectrogram, sample_rate=sample_rate, NFFT=NFFT_CHUNK_SIZE, nfilt=NUM_FILTER)

                # Mel-frequency Cepstral Coefficients
                if APPLY_MFCC:
                    mfcc_spectrum = mfcc(mel_spectrum)

                # Mean Normalization
                # reshape by subtract the freq vector with the mean of that band across each frame
                mel_spectrum = mel_spectrum - (np.mean(mel_spectrum, axis=1) + 1e-8).reshape((mel_spectrum.shape[0], 1))
                if APPLY_MFCC:
                    mfcc_spectrum -= (np.mean(mfcc_spectrum, axis=1) + 1e-8).reshape((mfcc_spectrum.shape[0], 1))

                #
                # Export to graph image
                #
                if EXPORT_IMAGE:
                    # generate file name
                    spec_img = _normalize_path(
                        '{prefix}/{id}-{feature}-1_spec.png'.format(prefix=PREFIX_OUTPUT, id=id, feature=feature))
                    mel_img = _normalize_path(
                        '{prefix}/{id}-{feature}-2_mel.png'.format(prefix=PREFIX_OUTPUT, id=id, feature=feature))
                    mfcc_img = _normalize_path(
                        '{prefix}/{id}-{feature}-3_mfcc.png'.format(prefix=PREFIX_OUTPUT, id=id, feature=feature))

                    spec(t, f, spectrogram, spec_img)  # show graph
                    mel_spec(t, np.arange(0, mel_spectrum.shape[0], 1) + 1, mel_spectrum, mel_img)  # show graph
                    if APPLY_MFCC:
                        mfcc_spec(t, np.arange(0, mfcc_spectrum.shape[0], 1) + 1, mfcc_spectrum, mfcc_img)  # show graph

                # Save data to list
                index_wav.setdefault(id, {})
                index_wav[id].setdefault(feature, np.ndarray)

                index_wav[id][feature] = mfcc_spectrum if APPLY_MFCC else mel_spectrum
                print('proceed: {:<30}{:<50}'.format(id, feature))

            except Exception as e:
                print(e)
                ban.append(id)

    for feat in key_col:
        # create features folder in raw path
        folder = _normalize_path('{prefix}/{feature}'.format(prefix=PREFIX_OUTPUT, feature=feat))
        if not os.path.exists(folder):
            os.mkdir(folder)

        # calculate the shape of each feature
        max_time_set = set()
        for id in index_wav:
            ban_by_feat.setdefault(feat, [])
            try:
                max_time_set.add(index_wav[id][feat].shape[1])
            except:
                ban_by_feat[feat].append(id)
        max_time = max(max_time_set)

        # start looping and export data into np file
        for id in index_wav:
            if id in ban_by_feat[feat]:
                continue

            # loop through each feature and save the files
            print('Padding wav ID {}'.format(id))
            wav = index_wav[id][feat]
            wav = np.pad(wav, ((0, 0), (0, max_time - wav.shape[1])))
            wav.reshape(wav.shape[0], wav.shape[1], 1)

            npz_raw = _normalize_path('{folder}/{id}.npz'.format(folder=folder, id=id))
            np.savez(npz_raw, wav)
            print(npz_raw)

    # export ban list if found
    print("ban list: {}".format(ban))
    print("ban by feature: {}".format(ban_by_feat))


if __name__ == '__main__':
    main()
