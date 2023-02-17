from typing import Any

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl


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
    def __init__(self, vocal_mask_model: VocalMask):
        super().__init__()
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

    def training_step(self, batch):
        x_spectrogram, binary_mask_true = batch

        binary_mask_prediction = self(x_spectrogram)
        loss = F.binary_cross_entropy_with_logits(
            input=binary_mask_prediction,
            target=binary_mask_true
        )

        self.log("train_loss", loss)

        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-3)

        return optimizer
