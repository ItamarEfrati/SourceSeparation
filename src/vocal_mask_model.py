from typing import Any

import librosa
import musdb
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
import torchaudio
from musdb import MultiTrack
from tqdm import tqdm

from utils.audio import spectrogram, inv_spectrogram, save_wav


class Conv3x3(nn.Module):
    def __init__(self, in_planes, out_planes, activation, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.activation = activation

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))


class FC(nn.Module):
    def __init__(self, indims, outdims, activation):
        super().__init__()
        self.fc = nn.Linear(indims, outdims)
        self.bn = nn.BatchNorm1d(outdims)
        self.activation = activation

    def forward(self, x):
        return self.activation(self.bn(self.fc(x)))


class VocalMask(nn.Module):
    def __init__(self, input_dims, output_dims):
        super().__init__()
        self.activation = nn.LeakyReLU()

        self.features_extractor = nn.Sequential(
            Conv3x3(1, 32, self.activation),
            Conv3x3(32, 16, self.activation),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.25),
            Conv3x3(16, 64, self.activation),
            Conv3x3(64, 16, self.activation),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.5),
            nn.Flatten()
        )

        fcdims = 16 * np.product([dim // 4 for dim in input_dims])

        self.mask_creator = nn.Sequential(
            FC(fcdims, 128, self.activation),
            nn.Dropout(p=0.5),
            nn.Linear(128, output_dims),
        )

    def forward(self, x):
        """

        :param x: batch sample of size (batch_size, 1, freq_bin_size, window_size)
        :type x: torch.Tensor
        :return:  sample binary vocal mask os size (batch_size, freq_bin_size)
        :rtype: torch.Tensor
        """
        features = self.features_extractor(x)
        mask_logits = self.mask_creator(features)

        return mask_logits


class LitVocalMask(pl.LightningModule):
    def __init__(self,
                 vocal_mask_model: VocalMask,
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
                 reconstruction_batch_size=512):
        super().__init__()
        self.save_hyperparameters(ignore=['vocal_mask_model'])
        self.vocal_mask_model = vocal_mask_model

    def forward(self, x):
        """

        :param x: batch sample of size (batch_size, 1, freq_bin_size, window_size)
        :type x: torch.Tensor
        :return:  sample binary vocal mask os size (batch_size, freq_bin_size)
        :rtype: torch.Tensor
        """
        mask_logits = self.vocal_mask_model(x)

        return mask_logits

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-3, weight_decay=0.3)

        return optimizer

    def training_step(self, batch, batch_idx):
        x_spectrogram, binary_mask_true = batch

        binary_mask_prediction = self(x_spectrogram)
        loss = F.binary_cross_entropy_with_logits(
            input=binary_mask_prediction,
            target=binary_mask_true,
            reduce=None
        )

        loss = loss.mean()

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x_spectrogram, binary_mask_true = batch

        binary_mask_prediction = self(x_spectrogram)

        loss = F.binary_cross_entropy_with_logits(
            input=binary_mask_prediction,
            target=binary_mask_true,
            reduce=None
        )

        loss = loss.mean()

        self.log("val_loss", loss)

        return loss

    def test_step(self, batch, batch_idx):
        x_spectrogram, binary_mask_true = batch

        binary_mask_prediction = self(x_spectrogram)

        loss = F.binary_cross_entropy_with_logits(
            input=binary_mask_prediction,
            target=binary_mask_true,
            reduce=None
        )

        loss = loss.mean()

        self.log("test_loss", loss)

        return loss
