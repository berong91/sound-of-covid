import os

import numpy as np
from matplotlib import pyplot as plt
from scipy.io import wavfile

from data_config import PREFIX_OUTPUT, BANNED_ID, key_col, EXPORT_IMAGE, EXTRACTED_DATA_PATH, APPLY_MFCC
from utils import _normalize_path, stft_spectrogram, mel_filter, mfcc


def spec(t: np.ndarray, f: np.ndarray, Zxx: np.ndarray, output: str) -> None:
    plt.clf()
    plt.pcolormesh(t, f, Zxx, vmin=0, vmax=2 * np.sqrt(2), shading='gouraud')
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    if output:
        plt.savefig(output)
    else:
        plt.show()


def mel_spec(t: np.ndarray, f: np.ndarray, Zxx: np.ndarray, output: str) -> None:
    plt.clf()
    plt.pcolormesh(t, f, Zxx, vmin=0, vmax=2 * np.sqrt(2), shading='gouraud')
    plt.title('Mel Log Spectrum')
    plt.ylabel('Mel Filter [KHz]')
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
    NFFT = 256
    num_filter = 40
    ban = []
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
                t, f, spectrogram = stft_spectrogram(sample_rate, audio, NFFT=NFFT)

                # Mel Filter Banks power spectrum
                mel_spectrum = mel_filter(spectrogram, sample_rate=sample_rate, NFFT=NFFT, nfilt=num_filter)

                # Mel-frequency Cepstral Coefficients
                if APPLY_MFCC:
                    mfcc_spectrum = mfcc(mel_spectrum)

                # Mean Normalization
                # reshape by subtract the freq vector with the mean of that band across each frame
                mel_spectrum = mel_spectrum - (np.mean(mel_spectrum, axis=1) + 1e-8).reshape((mel_spectrum.shape[0], 1))
                if APPLY_MFCC:
                    mfcc_spectrum -= (np.mean(mfcc_spectrum, axis=1) + 1e-8).reshape((mfcc_spectrum.shape[0], 1))
                    # mfcc_spectrum = mfcc_spectrum - (np.mean(mfcc_spectrum, axis=0) + 1e-8)

                # Export to graph image
                if EXPORT_IMAGE:
                    # generate file name
                    spec_img = _normalize_path('{}/{}-{}-1_spec.png'.format(PREFIX_OUTPUT, id, feature))
                    mel_img = _normalize_path('{}/{}-{}-2_mel.png'.format(PREFIX_OUTPUT, id, feature))
                    mfcc_img = _normalize_path('{}/{}-{}-3_mfcc.png'.format(PREFIX_OUTPUT, id, feature))

                    spec(t, f, spectrogram, spec_img)  # show graph
                    mel_spec(t, np.arange(0, mel_spectrum.shape[0], 1) + 1, mel_spectrum, mel_img)  # show graph
                    if APPLY_MFCC:
                        mfcc_spec(t, np.arange(0, mfcc_spectrum.shape[0], 1) + 1, mfcc_spectrum, mfcc_img)  # show graph

                # Save data to list
                index_wav.setdefault(id, np.ndarray)
                index_wav[id] = mfcc_spectrum if APPLY_MFCC else mel_spectrum
                print('proceed: {:<30}{:<50}'.format(id, feature))
            except Exception as e:
                print(e)
                ban.append(id)

    max_time = max(set(index_wav[id].shape[1] for id in index_wav))
    for id in index_wav:
        print('Padding wav ID {}'.format(id))
        wav = index_wav[id]
        wav = np.pad(wav, ((0, 0), (0, max_time - wav.shape[1])))
        wav.reshape(wav.shape[0], wav.shape[1], 1)

        npz_raw = _normalize_path('{}/{}.npz'.format(PREFIX_OUTPUT, id))
        print(npz_raw)
        np.savez(npz_raw, wav)

    # export ban list if found
    print("ban list: {}".format(ban))


if __name__ == '__main__':
    main()
