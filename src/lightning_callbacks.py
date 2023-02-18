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
from spectorgram_dataset import SpectrogramDataset, basic_collate, MyCollator
import copy
import librosa
import matplotlib.pyplot as plt
import librosa.display

from torchvision import transforms
from PIL import Image


def get_spec_fig(spec,
                 title: str,
                 y_axis: str = 'mel'):
    fig, ax = plt.subplots()
    img = librosa.display.specshow(spec, x_axis='time', y_axis=y_axis, ax=ax)
    ax.set(title=title)

    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)

    return buf


class LogVocalSeparationGridCallback(pl.Callback):
    def __init__(self, data_module: MUSDBDataModule,
                 training_spectrogram_id: int = 16,
                 every_n_epochs: int = 1,
                 every_n_steps: int = 1000):
        super().__init__()
        data_module.setup()
        train_dataset = data_module.train_data
        self.filtered_dataset = copy.deepcopy(train_dataset)

        self.specific_mix_spectrogram_array = torch.tensor(train_dataset.get_full_mix_spectrogram(training_spectrogram_id).squeeze(0))
        self.specific_vocal_spectrogram_array = torch.tensor(train_dataset.get_full_vocal_spectrogram(training_spectrogram_id))
        metadata_arr = np.array(data_module.train_data.metadata)
        spec_id_indices = np.argwhere(metadata_arr[:, 0] == f"spec{training_spectrogram_id:06d}").reshape(-1)
        filtered_metadata = [(t[0], int(t[1]), int(t[2])) for t in map(tuple, metadata_arr[spec_id_indices])]
        self.filtered_dataset.metadata = filtered_metadata

        # Only save those images every N epochs (otherwise tensorboard gets quite large)
        self.every_n_epochs = every_n_epochs
        self.every_n_steps = every_n_steps

    def add_images_to_tensor_board(self, trainer, masked_spectrogram):
        convert_tensor = transforms.ToTensor()
        original_spec = convert_tensor(
            Image.open(
                get_spec_fig(
                    spec=self.specific_mix_spectrogram_array.numpy(),
                    title="Mixture"
                )
            )
        )
        vocal_spec = convert_tensor(
            Image.open(
                get_spec_fig(
                    spec=self.specific_vocal_spectrogram_array.numpy(),
                    title="Vocal True"
                )
            )
        )
        predicted_vocal_spec = convert_tensor(
            Image.open(
                get_spec_fig(
                    spec=masked_spectrogram.numpy(),
                    title="Vocal Predicted"
                )
            )
        )

        imgs = torch.stack([original_spec, vocal_spec, predicted_vocal_spec])
        grid = torchvision.utils.make_grid(imgs, nrow=3)
        trainer.logger.experiment.add_image("Vocal Separation", grid, global_step=trainer.global_step)

    def get_masked_spectrogram(self,
                               pl_module,
                               threshold_value: float = 0.5):
        my_collator = MyCollator(threshold_value)
        data_loader = DataLoader(self.filtered_dataset, batch_size=512,
                                 collate_fn=my_collator, pin_memory=True)
        batch = []

        with torch.no_grad():
            # Reconstruct images
            for x, y in tqdm(data_loader):
                predicted_mask = pl_module(x)

                batch.append(predicted_mask)

        full_mask = torch.sigmoid(torch.vstack(batch).T)
        mask = full_mask > threshold_value
        full_mask[mask] = 1.
        full_mask[~mask] = 0.
        masked_spectrogram = full_mask * self.specific_mix_spectrogram_array[:, self.filtered_dataset.stft_frames:]

        return masked_spectrogram

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            pl_module.eval()

            masked_spectrogram = self.get_masked_spectrogram(pl_module)
            self.add_images_to_tensor_board(trainer, masked_spectrogram)

            pl_module.train()

    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule",
                           outputs: Union[Tensor, Dict[str, Any]], batch: Any, batch_idx: int):
        if batch_idx % self.every_n_steps == 0:
            pl_module.eval()

            masked_spectrogram = self.get_masked_spectrogram(pl_module)
            self.add_images_to_tensor_board(trainer, masked_spectrogram)

            pl_module.train()
