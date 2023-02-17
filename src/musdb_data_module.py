import os
import pickle

import pytorch_lightning as pl

from torch.utils.data import DataLoader
from src.spectorgram_dataset import SpectrogramDataset, basic_collate


class MUSDBDataModule(pl.LightningDataModule):
    def __init__(self,
                 data_dir: str = "../data/musdb18",
                 stft_frames=25,
                 stft_stride=1,
                 hop_size=256,
                 fft_size=1024,
                 mel_freqs=None,
                 fmin=20,
                 min_level_db=-100,
                 ref_level_db=20,
                 batch_size: int = 32,
                 train_mask_threshold=0.5,
                 test_mask_threshold=0.1):
        super().__init__()
        self.offset = stft_frames // 2
        self.save_hyperparameters()

    def _split_wav(self, wav_mix, wav_vocal, track_name):
        size = wav_mix.shape[0] - self.hparams.stft_frames
        for i in range(size, self.hparams.stft_stride):
            j = i + self.hparams.stft_frames
            x = wav_mix[:, :, i:j]
            y = wav_vocal[:, i + self.offset]

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str = None):
        with open(os.path.join(self.hparams.data_dir, 'spec_info.pkl'), 'rb') as f:
            train_specs = pickle.load(f)
        test_path = os.path.join(self.hparams.data_dir, "test")
        with open(os.path.join(test_path, "test_spec_info.pkl"), 'rb') as f:
            test_specs = pickle.load(f)

        test_path = os.path.join(self.hparams.data_dir, "test")
        self.train_data = SpectrogramDataset(self.hparams.data_dir, train_specs, self.hparams.stft_frames,
                                             self.hparams.stft_stride)
        self.test_data = SpectrogramDataset(test_path, test_specs, self.hparams.stft_frames, self.hparams.stft_stride)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.hparams.batch_size,
                          collate_fn=lambda b: basic_collate(b, self.hparams.train_mask_threshold))

    # def val_dataloader(self):
    #     return DataLoader(self.val_data, batch_size=self.hparams.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.hparams.batch_size,
                          collate_fn=lambda b: basic_collate(b, self.hparams.test_mask_threshold))



