from typing import Any

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl


class VocalMask(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """

        :param x: batch sample of size (batch_size, 1, freq_bin_size, window_size)
        :type x: torch.Tensor
        :return:  sample binary vocal mask os size (batch_size, freq_bin_size)
        :rtype: torch.Tensor
        """
        pass


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
