import os
from io import BytesIO
from typing import Any, Union, Dict

import numpy as np
import torch
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from musdb_data_module import MUSDBDataModule
from spectorgram_dataset import SpectrogramDataset, MyCollator
import copy
import librosa
import matplotlib.pyplot as plt
import librosa.display

from torchvision import transforms
from PIL import Image

from utils.audio import spectrogram, inv_spectrogram
from mir_eval import separation


def get_spec_fig(spec,
                 title: str,
                 y_axis: str = 'mel'):
    fig, ax = plt.subplots()
    img = librosa.display.specshow(spec, x_axis='time', y_axis=y_axis, ax=ax)
    ax.set(title=title)

    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf


class LogVocalSeparationGridCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.data_dir = '../data/musdb18/preprocess'
        self.train_spec = os.listdir(rf'{self.data_dir}/train/spec_mix')[0]
        self.val_spec = os.listdir(rf'{self.data_dir}/val/spec_mix')[0]
        self.test_spec = os.listdir(rf'{self.data_dir}/test/spec_mix')[0]

    def _load_specs(self, stage):
        spec = self.train_spec
        if stage == 'val':
            spec = self.val_spec
        if stage == 'test':
            spec = self.test_spec
        mel_spec_mix = np.load(os.path.join(self.data_dir, f'{stage}/spec_mix', spec), mmap_mode='r')
        mel_spec_vox = np.load(os.path.join(self.data_dir, f'{stage}/spec_vox', spec), mmap_mode='r')

        return mel_spec_mix, mel_spec_vox

    @torch.no_grad()
    def _get_mask(self, mel_spec, pl_module, threshold=0.5):
        padding = pl_module.hparams.stft_frames // 2
        mel_spec = np.pad(mel_spec, ((0, 0), (padding, padding)), 'constant', constant_values=0)
        window = pl_module.hparams.stft_frames
        size = mel_spec.shape[1]
        mask = []
        end = size - window
        for i in tqdm(range(0, end + 1, pl_module.hparams.reconstruction_batch_size)):
            x = [mel_spec[:, j:j + window] for j in range(i, i + pl_module.hparams.reconstruction_batch_size) if
                 j <= end]
            x = np.stack(x)
            _x = torch.FloatTensor(x[:, np.newaxis, :, :]).to(pl_module.device)
            _y = pl_module.forward(_x)
            y = _y.to(torch.device('cpu')).detach().numpy()
            mask += [y[j] for j in range(y.shape[0])]
        mask = np.vstack(mask).T
        return mask > threshold

    @staticmethod
    def add_images_to_tensor_board(trainer, mel_spec_mix, mel_spec_vox, masked_spectrogram, stage):
        convert_tensor = transforms.ToTensor()
        original_spec = convert_tensor(
            Image.open(
                get_spec_fig(
                    spec=mel_spec_mix,
                    title="Mixture"
                )
            )
        )
        vocal_spec = convert_tensor(
            Image.open(
                get_spec_fig(
                    spec=mel_spec_vox,
                    title="Vocal True"
                )
            )
        )
        predicted_vocal_spec = convert_tensor(
            Image.open(
                get_spec_fig(
                    spec=masked_spectrogram,
                    title="Vocal Predicted"
                )
            )
        )

        imgs = torch.stack([original_spec, vocal_spec, predicted_vocal_spec])
        grid = torchvision.utils.make_grid(imgs, nrow=3)
        trainer.logger.experiment.add_image(f"{stage} Vocal Separation", grid, global_step=trainer.global_step)

    def _get_masked_spectrogram(self,
                                mel_spec_mix,
                                pl_module,
                                threshold_value: float = 0.5):
        full_mask = self._get_mask(mel_spec_mix, pl_module)
        mask = full_mask > threshold_value
        full_mask[mask] = 1.
        full_mask[~mask] = 0.
        masked_spectrogram = full_mask * mel_spec_mix

        return masked_spectrogram

    def on_train_epoch_end(self, trainer, pl_module):
        is_training = pl_module.training
        pl_module.eval()
        mel_spec_mix, mel_spec_vox = self._load_specs('train')
        mel_spec_mix = mel_spec_mix.squeeze(0)
        masked_spectrogram = self._get_masked_spectrogram(mel_spec_mix, pl_module)
        self.add_images_to_tensor_board(trainer, mel_spec_mix, mel_spec_vox, masked_spectrogram, 'train')
        if is_training:
            pl_module.train()

    def on_validation_epoch_end(self, trainer, pl_module):
        is_training = pl_module.training
        pl_module.eval()
        mel_spec_mix, mel_spec_vox = self._load_specs('val')
        mel_spec_mix = mel_spec_mix.squeeze(0)
        masked_spectrogram = self._get_masked_spectrogram(mel_spec_mix, pl_module)
        self.add_images_to_tensor_board(trainer, mel_spec_mix, mel_spec_vox, masked_spectrogram, 'val')
        if is_training:
            pl_module.train()

    def on_test_epoch_end(self, trainer, pl_module):
        is_training = pl_module.training
        pl_module.eval()
        mel_spec_mix, mel_spec_vox = self._load_specs('test')
        mel_spec_mix = mel_spec_mix.squeeze(0)
        masked_spectrogram = self._get_masked_spectrogram(mel_spec_mix, pl_module)
        self.add_images_to_tensor_board(trainer, mel_spec_mix, mel_spec_vox, masked_spectrogram, 'test')
        if is_training:
            pl_module.train()


class LogAudio(pl.Callback):
    def __init__(self, every_n_epochs: int = 5):
        super().__init__()

        # Only save those images every N epochs (otherwise tensorboard gets quite large)
        self.every_n_epochs = every_n_epochs
        self.data_dir = '..\data\musdb18\preprocess'
        self.train_track = os.listdir(rf'{self.data_dir}\train\wav_mix')[0]
        self.val_track = os.listdir(rf'{self.data_dir}\val\wav_mix')[0]
        self.test_track = os.listdir(rf'{self.data_dir}\test\wav_mix')[0]

    def _load_wavs(self, stage, pl_module):
        track = self.train_track
        if stage == 'val':
            track = self.val_track
        if stage == 'test':
            track = self.test_track
        wav, sr = librosa.load(os.path.join(self.data_dir, f'{stage}\wav_mix', track),
                               sr=pl_module.hparams.sample_rate)

        vox, sr = librosa.load(os.path.join(self.data_dir, f'{stage}\wav_vox', track),
                               sr=pl_module.hparams.sample_rate)
        return wav, vox

    @torch.no_grad()
    def _get_mask(self, wav, pl_module, threshold=0.5):
        _mel_spec, stft = spectrogram(wav,
                                      power=pl_module.hparams.mix_power_factor,
                                      hop_size=pl_module.hparams.hop_size,
                                      fft_size=pl_module.hparams.fft_size,
                                      fmin=pl_module.hparams.fmin,
                                      ref_level_db=pl_module.hparams.ref_level_db,
                                      mel_freqs=pl_module.hparams.mel_freqs,
                                      min_level_db=pl_module.hparams.min_level_db)

        padding = pl_module.hparams.stft_frames // 2
        mel_spec = np.pad(_mel_spec, ((0, 0), (padding, padding)), 'constant', constant_values=0)
        window = pl_module.hparams.stft_frames
        size = mel_spec.shape[1]
        mask = []
        end = size - window
        for i in tqdm(range(0, end + 1, pl_module.hparams.reconstruction_batch_size)):
            x = [mel_spec[:, j:j + window] for j in range(i, i + pl_module.hparams.reconstruction_batch_size) if
                 j <= end]
            x = np.stack(x)
            _x = torch.FloatTensor(x[:, np.newaxis, :, :]).to(pl_module.device)
            _y = pl_module.forward(_x)
            y = _y.to(torch.device('cpu')).detach().numpy()
            mask += [y[j] for j in range(y.shape[0])]
        mask = np.vstack(mask).T
        return mask > threshold, stft

    def _log_audio(self, pl_module, trainer, stage):

        wav, vox = self._load_wavs(stage=stage, pl_module=pl_module)

        vox_mask, stft = self._get_mask(wav, pl_module)

        estimates = inv_spectrogram(stft * vox_mask, pl_module.hparams.hop_size)
        original = librosa.istft(stft, hop_length=pl_module.hparams.hop_size)
        if pl_module.hparams.original_sample_rate != pl_module.hparams.sample_rate:
            estimates = librosa.resample(estimates, orig_sr=pl_module.hparams.sample_rate,
                                         target_sr=pl_module.hparams.original_sample_rate)
            original = librosa.resample(original, orig_sr=pl_module.hparams.sample_rate,
                                        target_sr=pl_module.hparams.original_sample_rate)
            vox = librosa.resample(vox, orig_sr=pl_module.hparams.sample_rate,
                                   target_sr=pl_module.hparams.original_sample_rate)

        trainer.logger.experiment.add_audio(f'{stage} target vocal', vox, trainer.global_step,
                                            pl_module.hparams.original_sample_rate)
        trainer.logger.experiment.add_audio(f'{stage} target mix', original, trainer.global_step,
                                            pl_module.hparams.original_sample_rate)
        trainer.logger.experiment.add_audio(f'{stage} predicted vocal', estimates, trainer.global_step,
                                            pl_module.hparams.original_sample_rate)

    def on_train_epoch_end(self, trainer, pl_module):
        is_training = pl_module.training
        pl_module.eval()
        self._log_audio(pl_module, trainer, stage='train')
        if is_training:
            pl_module.train()

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        is_training = pl_module.training
        pl_module.eval()
        self._log_audio(pl_module, trainer, stage='val')
        if is_training:
            pl_module.train()

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        is_training = pl_module.training
        pl_module.eval()
        self._log_audio(pl_module, trainer, stage='test')
        if is_training:
            pl_module.train()


class LogEvaluationMetrics(pl.Callback):
    def __init__(self, every_n_epochs: int = 5):
        super().__init__()

        # Only save those images every N epochs (otherwise tensorboard gets quite large)
        self.every_n_epochs = every_n_epochs
        self.data_dir = '..\data\musdb18\preprocess'
        self.train_track = os.listdir(rf'{self.data_dir}\train\wav_mix')[0]
        self.val_track = os.listdir(rf'{self.data_dir}\val\wav_mix')[0]
        self.test_track = os.listdir(rf'{self.data_dir}\test\wav_mix')[0]

    def _load_wavs(self, stage, pl_module):
        track = self.train_track
        if stage == 'val':
            track = self.val_track
        if stage == 'test':
            track = self.test_track
        wav, sr = librosa.load(os.path.join(self.data_dir, f'{stage}\wav_mix', track),
                               sr=pl_module.hparams.sample_rate)

        vox, sr = librosa.load(os.path.join(self.data_dir, f'{stage}\wav_vox', track),
                               sr=pl_module.hparams.sample_rate)
        return wav, vox

    @torch.no_grad()
    def _get_mask(self, wav, pl_module, threshold=0.5):
        _mel_spec, stft = spectrogram(wav,
                                      power=pl_module.hparams.mix_power_factor,
                                      hop_size=pl_module.hparams.hop_size,
                                      fft_size=pl_module.hparams.fft_size,
                                      fmin=pl_module.hparams.fmin,
                                      ref_level_db=pl_module.hparams.ref_level_db,
                                      mel_freqs=pl_module.hparams.mel_freqs,
                                      min_level_db=pl_module.hparams.min_level_db)

        padding = pl_module.hparams.stft_frames // 2
        mel_spec = np.pad(_mel_spec, ((0, 0), (padding, padding)), 'constant', constant_values=0)
        window = pl_module.hparams.stft_frames
        size = mel_spec.shape[1]
        mask = []
        end = size - window
        for i in tqdm(range(0, end + 1, pl_module.hparams.reconstruction_batch_size)):
            x = [mel_spec[:, j:j + window] for j in range(i, i + pl_module.hparams.reconstruction_batch_size) if
                 j <= end]
            x = np.stack(x)
            _x = torch.FloatTensor(x[:, np.newaxis, :, :]).to(pl_module.device)
            _y = pl_module.forward(_x)
            y = _y.to(torch.device('cpu')).detach().numpy()
            mask += [y[j] for j in range(y.shape[0])]
        mask = np.vstack(mask).T
        return mask > threshold, stft

    def _evaluate_metrics(self,
                          original_source: np.ndarray,
                          predicted_source: np.ndarray):
        sdr, sir, sar, _ = separation.bss_eval_sources(
          original_source,
          predicted_source)

        return sdr.mean(), sir.mean(), sar.mean()

    def _log_metrics(self, pl_module, trainer, stage):

        wav, vox = self._load_wavs(stage=stage, pl_module=pl_module)
        vox_mask, stft = self._get_mask(wav, pl_module)
        estimates = inv_spectrogram(stft * vox_mask, pl_module.hparams.hop_size)

        if pl_module.hparams.original_sample_rate != pl_module.hparams.sample_rate:
            estimates = librosa.resample(estimates, orig_sr=pl_module.hparams.sample_rate,
                                         target_sr=pl_module.hparams.original_sample_rate)
            vox = librosa.resample(vox, orig_sr=pl_module.hparams.sample_rate,
                                   target_sr=pl_module.hparams.original_sample_rate)

        sdr, sir, sar = self._evaluate_metrics(vox, estimates)

        trainer.logger.add_scalar(
            tag=f"{stage} sdr",
            scalar_value=sdr,
            global_step=trainer.global_step
        )
        trainer.logger.add_scalar(
            tag=f"{stage} sir",
            scalar_value=sir,
            global_step=trainer.global_step
        )
        trainer.logger.add_scalar(
            tag=f"{stage} sar",
            scalar_value=sar,
            global_step=trainer.global_step
        )

    def on_train_epoch_end(self, trainer, pl_module):
        is_training = pl_module.training
        pl_module.eval()
        self._log_metrics(pl_module, trainer, stage='train')
        if is_training:
            pl_module.train()

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        is_training = pl_module.training
        pl_module.eval()
        self._log_metrics(pl_module, trainer, stage='val')
        if is_training:
            pl_module.train()

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        is_training = pl_module.training
        pl_module.eval()
        self._log_metrics(pl_module, trainer, stage='test')
        if is_training:
            pl_module.train()