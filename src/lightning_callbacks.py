import os
from collections import defaultdict
from io import BytesIO

import numpy as np
import torch
import torchvision
import pytorch_lightning as pl
from tqdm import tqdm

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


class LogAll(pl.Callback):
    def __init__(self):
        super().__init__()

        # Only save those images every N epochs (otherwise tensorboard gets quite large)
        self.step = None
        self.is_first = defaultdict(lambda: True)
        self.data_dir = '..\data\musdb18\preprocess'
        self.train_tracks = os.listdir(rf'{self.data_dir}\train\wav_mix')
        self.val_tracks = os.listdir(rf'{self.data_dir}\val\wav_mix')
        self.test_tracks = os.listdir(rf'{self.data_dir}\test\wav_mix')

    # region Audio functions
    def _load_wavs(self, stage, pl_module, track):
        wav, sr = librosa.load(os.path.join(self.data_dir, f'{stage}\wav_mix', track),
                               sr=pl_module.hparams.sample_rate)

        vox, sr = librosa.load(os.path.join(self.data_dir, f'{stage}\wav_vox', track),
                               sr=pl_module.hparams.sample_rate)
        return wav, vox

    def _get_tracks(self, stage):
        tracks = self.train_tracks
        if stage == 'val':
            tracks = self.val_tracks
        if stage == 'test':
            tracks = self.test_tracks
        return tracks

    @torch.no_grad()
    def _get_mask(self, mel_spec, pl_module):

        padding = pl_module.hparams.stft_frames // 2
        mel_spec = np.pad(mel_spec, ((0, 0), (padding, padding)), 'constant', constant_values=0)
        window = pl_module.hparams.stft_frames
        size = mel_spec.shape[1]
        mask = []
        end = size - window
        for i in range(0, end + 1, pl_module.hparams.reconstruction_batch_size):
            x = [mel_spec[:, j:j + window] for j in range(i, i + pl_module.hparams.reconstruction_batch_size) if
                 j <= end]
            x = np.stack(x)
            _x = torch.FloatTensor(x[:, np.newaxis, :, :]).to(pl_module.device)
            _y = torch.sigmoid(pl_module(_x))
            y = _y.to(torch.device('cpu')).detach().numpy()
            mask += [y[j] for j in range(y.shape[0])]
        mask = np.vstack(mask).T
        return mask

    def _get_masked_spectrogram(self,
                                wav,
                                vox,
                                pl_module,
                                threshold_value: float = 0.1):
        mel_spec_mix, spec_mix = spectrogram(wav,
                                             power=pl_module.hparams.mix_power_factor,
                                             hop_size=pl_module.hparams.hop_size,
                                             fft_size=pl_module.hparams.fft_size,
                                             fmin=pl_module.hparams.fmin,
                                             ref_level_db=pl_module.hparams.ref_level_db,
                                             mel_freqs=pl_module.hparams.mel_freqs,
                                             min_level_db=pl_module.hparams.min_level_db)

        mel_spec_vox, _ = spectrogram(vox,
                                      power=pl_module.hparams.mix_power_factor,
                                      hop_size=pl_module.hparams.hop_size,
                                      fft_size=pl_module.hparams.fft_size,
                                      fmin=pl_module.hparams.fmin,
                                      ref_level_db=pl_module.hparams.ref_level_db,
                                      mel_freqs=pl_module.hparams.mel_freqs,
                                      min_level_db=pl_module.hparams.min_level_db)
        full_mask = self._get_mask(mel_spec_mix, pl_module)
        mask = full_mask > threshold_value
        full_mask[mask] = 1.
        full_mask[~mask] = 0.

        return mel_spec_mix, mel_spec_vox, spec_mix, full_mask

    @staticmethod
    def _evaluate_metrics(original_source: np.ndarray, predicted_source: np.ndarray):
        if sum(predicted_source) == 0:
            predicted_source[0] = 1
        sdr, sir, sar, _ = separation.bss_eval_sources(original_source, predicted_source)

        return sdr.mean(), sir.mean(), sar.mean()

    # endregion

    # region Logs
    def add_images_to_tensor_board(self, logger, mel_spec_mix, mel_spec_vox, full_mask, stage):
        convert_tensor = transforms.ToTensor()
        # if self.is_first[stage]:
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
                    spec=full_mask,
                    title="Vocal Mask"
                )
            )
        )

        imgs = torch.stack([original_spec, vocal_spec, predicted_vocal_spec])
        grid = torchvision.utils.make_grid(imgs, nrow=3)
        logger.experiment.add_image(f"{stage} Vocal Separation", grid, global_step=self.step)

    def _log_audio(self, pl_module, logger, wav, vox, estimates, stage):
        if pl_module.hparams.original_sample_rate != pl_module.hparams.sample_rate:
            estimates = librosa.resample(estimates, orig_sr=pl_module.hparams.sample_rate,
                                         target_sr=pl_module.hparams.original_sample_rate)
            wav = librosa.resample(wav, orig_sr=pl_module.hparams.sample_rate,
                                   target_sr=pl_module.hparams.original_sample_rate)
            vox = librosa.resample(vox, orig_sr=pl_module.hparams.sample_rate,
                                   target_sr=pl_module.hparams.original_sample_rate)

        audio_length = pl_module.hparams.original_sample_rate * 30

        if self.is_first[stage]:
            logger.experiment.add_audio(f'{stage} target vocal', vox[:audio_length], self.step,
                                        pl_module.hparams.original_sample_rate)
            logger.experiment.add_audio(f'{stage} target mix', wav[:audio_length], self.step,
                                        pl_module.hparams.original_sample_rate)
            self.is_first[stage] = False
        logger.experiment.add_audio(f'{stage} predicted vocal', estimates[:audio_length], self.step,
                                    pl_module.hparams.original_sample_rate)

    def _log_metrics(self, pl_module, logger, tracks, stage):
        sdr, sir, sar = [], [], []
        for track in tqdm(tracks):
            wav, vox = self._load_wavs(stage=stage, pl_module=pl_module, track=track)
            mel_spec_mix, mel_spec_vox, spec_mix, full_mask = self._get_masked_spectrogram(wav=wav, vox=vox,
                                                                                           pl_module=pl_module)
            masked_spectrogram = full_mask * spec_mix
            estimates = inv_spectrogram(masked_spectrogram, pl_module.hparams.hop_size, length=len(vox))

            a, b, c = self._evaluate_metrics(vox, estimates)
            sdr.append(a)
            sir.append(b)
            sar.append(c)

        logger.experiment.add_scalar(
            tag=f"{stage} sdr",
            scalar_value=np.mean(sdr),
            global_step=self.step
        )
        logger.experiment.add_scalar(
            tag=f"{stage} sir",
            scalar_value=np.mean(sir),
            global_step=self.step
        )
        logger.experiment.add_scalar(
            tag=f"{stage} sar",
            scalar_value=np.mean(sar),
            global_step=self.step
        )

        return wav, vox, estimates, mel_spec_mix, mel_spec_vox, full_mask, masked_spectrogram, np.mean(sdr)

    # endregion

    def eval_step(self, logger, pl_module, step, stage):
        self.step = step
        is_training = pl_module.training
        pl_module.eval()
        tracks = self._get_tracks(stage)
        wav, vox, estimates, mel_spec_mix, mel_spec_vox, full_mask, masked_spectrogram, sdr = \
            self._log_metrics(pl_module, logger, tracks=tracks, stage=stage)
        self.add_images_to_tensor_board(logger, mel_spec_mix, mel_spec_vox, full_mask, stage)
        self._log_audio(pl_module, logger, wav=wav, vox=vox, estimates=estimates, stage=stage)

        if is_training:
            pl_module.train()

        return sdr

    def on_train_epoch_end(self, trainer, pl_module):
        is_training = pl_module.training
        pl_module.eval()
        wav, vox, estimates, mel_spec_mix, mel_spec_vox, full_mask, masked_spectrogram = \
            self._log_metrics(pl_module, trainer.logger, tracks=self.train_tracks, stage='train')
        self.add_images_to_tensor_board(trainer.logger, mel_spec_mix, mel_spec_vox, full_mask, 'train')
        self._log_audio(pl_module, trainer.logger, wav=wav, vox=vox, estimates=estimates, stage='train')

        if is_training:
            pl_module.train()

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        is_training = pl_module.training
        pl_module.eval()
        wav, vox, estimates, mel_spec_mix, mel_spec_vox, full_mask, masked_spectrogram = \
            self._log_metrics(pl_module, trainer.logger, tracks=self.train_tracks, stage='val')
        self.add_images_to_tensor_board(trainer.logger, mel_spec_mix, mel_spec_vox, full_mask, 'val')
        self._log_audio(pl_module, trainer.logger, wav=wav, vox=vox, estimates=estimates, stage='val')

        if is_training:
            pl_module.train()

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        is_training = pl_module.training
        pl_module.eval()
        wav, vox, estimates, mel_spec_mix, mel_spec_vox, full_mask, masked_spectrogram = \
            self._log_metrics(pl_module, trainer.logger, tracks=self.train_tracks, stage='test')
        self.add_images_to_tensor_board(trainer.logger, mel_spec_mix, mel_spec_vox, full_mask, 'test')
        self._log_audio(pl_module, trainer.logger, wav=wav, vox=vox, estimates=estimates, stage='test')

        if is_training:
            pl_module.train()
