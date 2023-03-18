import librosa
import numpy as np
import soundfile as sf


def load_wav(path, sample_rate):
    wav = librosa.load(path, sr=sample_rate)[0]
    return wav


def get_wav_slices(wav, window, stride):
    N = len(wav)
    return [(i, i + window) for i in range(0, N - window, stride)]


def save_wav(wav, path, sample_rate=None):
    sf.write(file=path, data=wav, samplerate=sample_rate)


def amp_to_db(x, min_level_db):
    min_level = np.exp(min_level_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))


def db_to_amp(x):
    return np.power(10.0, x * 0.05)


def normalize(S, min_level_db):
    return np.clip((S - min_level_db) / -min_level_db, 0, 1)


def denormalize(S, min_level_db):
    return (np.clip(S, 0, 1) * -min_level_db) + min_level_db


def preemphasis(x):
    from nnmnkwii.preprocessing import preemphasis
    return preemphasis(x)


def inv_preemphasis(x):
    from nnmnkwii.preprocessing import inv_preemphasis
    return inv_preemphasis(x)


def spectrogram(wav, power, hop_size, fft_size, fmin, ref_level_db, mel_freqs, min_level_db):
    stftS = librosa.stft(wav, n_fft=fft_size, hop_length=hop_size)
    wav = preemphasis(wav)
    S = librosa.stft(wav, n_fft=fft_size, hop_length=hop_size)
    if mel_freqs is None:
        mel_freqs = librosa.mel_frequencies(S.shape[0], fmin=fmin)
    _S = librosa.perceptual_weighting(np.abs(S) ** power, mel_freqs, ref=ref_level_db)
    return normalize(_S - ref_level_db, min_level_db), stftS


def inv_spectrogram(S, hop_size, length=None):
    y = librosa.istft(S, hop_length=hop_size, length=length)
    return inv_preemphasis(y)
