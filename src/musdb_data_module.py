import os
import pickle
import random
import zipfile

import librosa
import musdb
import numpy as np
import pytorch_lightning as pl

from multiprocessing import Manager, Process

import requests
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.spectorgram_dataset import SpectrogramDataset, basic_collate
from src.utils.audio import spectrogram, get_wav_slices, save_wav

REGULAR_URL = 'https://zenodo.org/record/1117372/files/musdb18.zip?download=1'
HQ_URL = 'https://zenodo.org/record/3338373/files/musdb18hq.zip?download=1'


class MUSDBDataModule(pl.LightningDataModule):
    def __init__(self,
                 data_dir: str = "data\musdb18",
                 download_url=REGULAR_URL,
                 num_workers=14,
                 original_sample_rate=44100,
                 sample_rate=22050,
                 mix_power_factor=2,
                 hop_size=256,
                 fft_size=1024,
                 fmin=20,
                 ref_level_db=20,
                 mel_freqs=None,
                 min_level_db=-100,
                 stft_frames=25,
                 stft_stride=1,
                 train_length=1024,
                 batch_size: int = 32,
                 train_mask_threshold=0.5,
                 test_mask_threshold=0.1):
        super().__init__()
        self.offset = stft_frames // 2
        self.train_length = sample_rate * 2
        self.save_hyperparameters()
        self.paths = {}

    # region Private
    def _init_dirs(self):
        output_dir = os.path.join(self.hparams.data_dir, 'preprocess')
        os.makedirs(output_dir, exist_ok=True)
        self.paths['train_path'] = os.path.join(output_dir, "train")
        self.paths['train_spect_mixture_path'] = os.path.join(self.paths['train_path'], "spec_mix")
        self.paths['train_spect_vocal_path'] = os.path.join(self.paths['train_path'], "spec_vox")
        self.paths['val_path'] = os.path.join(output_dir, "val")
        self.paths['val_spect_mixture_path'] = os.path.join(self.paths['val_path'], "spec_mix")
        self.paths['val_spect_vocal_path'] = os.path.join(self.paths['val_path'], "spec_vox")
        self.paths['val_wav_mix_path'] = os.path.join(self.paths['val_path'], "wav_mix")
        self.paths['val_wav_vox_path'] = os.path.join(self.paths['val_path'], "wav_vox")
        self.paths['test_path'] = os.path.join(output_dir, "test")
        self.paths['test_spect_mixture_path'] = os.path.join(self.paths['test_path'], "spec_mix")
        self.paths['test_spect_vocal_path'] = os.path.join(self.paths['test_path'], "spec_vox")
        self.paths['test_wav_mix_path'] = os.path.join(self.paths['test_path'], "wav_mix")
        self.paths['test_wav_vox_path'] = os.path.join(self.paths['test_path'], "wav_vox")
        for k, v in self.paths.items():
            os.makedirs(v, exist_ok=True)

    def _load_sample(self, track):
        audio = track.audio

        audio = librosa.to_mono(audio.T)
        if self.hparams.original_sample_rate != self.hparams.sample_rate:
            audio = librosa.resample(audio, orig_sr=self.hparams.original_sample_rate,
                                     target_sr=self.hparams.sample_rate)
        return audio

    def _load_samples(self, track):
        vocal_track = track.targets['vocals']
        mixture = self._load_sample(track)
        vocal = self._load_sample(vocal_track)
        return mixture, vocal

    def _generate_specs(self, L, track, idx, mixture_path, vocal_path):
        mixture, vocal = self._load_samples(track)
        mix_spec = spectrogram(mixture,
                               power=self.hparams.mix_power_factor,
                               hop_size=self.hparams.hop_size,
                               fft_size=self.hparams.fft_size,
                               fmin=self.hparams.fmin,
                               ref_level_db=self.hparams.ref_level_db,
                               mel_freqs=self.hparams.mel_freqs,
                               min_level_db=self.hparams.min_level_db)[0][np.newaxis, :, :]

        vox_spec = spectrogram(vocal,
                               power=self.hparams.mix_power_factor,
                               hop_size=self.hparams.hop_size,
                               fft_size=self.hparams.fft_size,
                               fmin=self.hparams.fmin,
                               ref_level_db=self.hparams.ref_level_db,
                               mel_freqs=self.hparams.mel_freqs,
                               min_level_db=self.hparams.min_level_db)[0]

        spec_id = f"spec{idx:06d}"
        L.append((spec_id, mix_spec.shape[2]))
        np.save(os.path.join(mixture_path, spec_id + ".npy"), mix_spec)
        np.save(os.path.join(vocal_path, spec_id + ".npy"), vox_spec)

    def _create_spectrogram(self, tracks, mix_path, vox_path, save_path):
        with Manager() as manager:
            spec_info = manager.list()
            processes = []
            for idx, track in enumerate(tqdm(tracks)):
                if len(processes) >= self.hparams.num_workers:
                    for p in processes:
                        p.join()
                    processes = []

                p = Process(target=self._generate_specs, args=(spec_info, track, idx, mix_path, vox_path))
                p.start()
                processes.append(p)
            if len(processes) > 0:
                for p in processes:
                    p.join()
            spec_info = list(spec_info)
            with open(save_path, 'wb') as f:
                random.shuffle(spec_info)
                pickle.dump(spec_info, f)

    def _create_slices_for_evaluation(self, tracks, stage):
        print(f"slicing {stage} samples")
        i = 0
        for track in tqdm(tracks):
            mixture, vocal = self._load_samples(track)
            slices = get_wav_slices(mixture, window=self.train_length, stride=self.train_length)
            for j, k in slices:
                # skip slices with no audio content
                if np.sum(vocal[j:k]) == 0:
                    continue
                file_name = f"{i:06d}.wav"
                save_wav(mixture[j:k],
                         os.path.join(self.paths[f'{stage}_wav_mix_path'], file_name),
                         sample_rate=self.hparams.original_sample_rate)
                save_wav(vocal[j:k],
                         os.path.join(self.paths[f'{stage}_wav_vox_path'], file_name),
                         self.hparams.original_sample_rate)
                i += 1

    def _download(self):
        headers = {'user-agent': 'Wget/1.16 (linux-gnu)'}
        response = requests.get(self.hparams.download_url, stream=True, headers=headers)
        content_length = int(response.headers['Content-Length'])
        pbar = tqdm(total=content_length)
        zip_file_path = os.path.join(self.hparams.data_dir, 'temp.zip')
        with open(zip_file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=30_000_000):
                if chunk:
                    f.write(chunk)
                pbar.update(len(chunk))
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(self.hparams.data_dir)

    # endregion

    def prepare_data(self) -> None:
        self._init_dirs()
        if os.path.exists(os.path.join(self.hparams.data_dir, 'preprocess')):
            return
        download = not os.path.exists(os.path.join(self.hparams.data_dir, 'train'))
        if download:
            self._download()
        dataset = musdb.DB(root=self.hparams.data_dir, is_wav=False)
        train_tracks = dataset.load_mus_tracks(subsets=['train'], split='train')
        val_tracks = dataset.load_mus_tracks(subsets=['train'], split='valid')
        test_tracks = dataset.load_mus_tracks(subsets=['test'])

        train_info_path = os.path.join(self.paths['train_path'], "spec_info.pkl")
        self._create_spectrogram(train_tracks, self.paths['train_spect_mixture_path'],
                                 self.paths['train_spect_vocal_path'], train_info_path)

        val_info_path = os.path.join(self.paths['val_path'], "spec_info.pkl")
        self._create_spectrogram(val_tracks, self.paths['val_spect_mixture_path'],
                                 self.paths['val_spect_vocal_path'], val_info_path)

        test_info_path = os.path.join(self.paths['test_path'], "spec_info.pkl")
        self._create_spectrogram(test_tracks, self.paths['test_spect_mixture_path'],
                                 self.paths['test_spect_vocal_path'], test_info_path)

        self._create_slices_for_evaluation(val_tracks, 'val')
        self._create_slices_for_evaluation(test_tracks, 'test')

    def setup(self, stage: str = None):
        self._init_dirs()
        with open(os.path.join(self.paths['train_path'], 'spec_info.pkl'), 'rb') as f:
            train_specs = pickle.load(f)
        with open(os.path.join(self.paths['val_path'], 'spec_info.pkl'), 'rb') as f:
            val_specs = pickle.load(f)
        with open(os.path.join(self.paths['test_path'], 'spec_info.pkl'), 'rb') as f:
            test_specs = pickle.load(f)

        self.train_data = SpectrogramDataset(self.paths['train_path'], train_specs, self.hparams.stft_frames,
                                             self.hparams.stft_stride)
        self.val_data = SpectrogramDataset(self.paths['val_path'], val_specs, self.hparams.stft_frames,
                                           self.hparams.stft_stride)
        self.test_data = SpectrogramDataset(self.paths['test_path'], test_specs, self.hparams.stft_frames,
                                            self.hparams.stft_stride)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.hparams.batch_size,
                          collate_fn=lambda b: basic_collate(b, self.hparams.train_mask_threshold))

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.hparams.batch_size,
                          collate_fn=lambda b: basic_collate(b, self.hparams.train_mask_threshold))

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.hparams.batch_size,
                          collate_fn=lambda b: basic_collate(b, self.hparams.test_mask_threshold))
