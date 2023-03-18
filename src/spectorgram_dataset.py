import os
import numpy as np
import torch

from torch.utils.data import Dataset
from tqdm import tqdm


class SpectrogramDataset(Dataset):
    def __init__(self, data_path, spectrogram_info, stft_frames, stft_stride, stage):
        self.data_path = data_path
        self.mix_path = os.path.join(data_path, "spec_mix")
        self.vox_path = os.path.join(data_path, "spec_vox")
        self.stft_frames = stft_frames
        self.stft_stride = stft_stride
        self.offset = stft_frames // 2
        self.metadata = self.get_slices(spectrogram_info, stage)

    def get_slices(self, spectrogram_info, stage):
        metadata = []
        print(f"Preparing {stage} dataset")
        for spectrogram in tqdm(spectrogram_info):
            size = spectrogram[1] - self.stft_frames
            for i in range(0, size, self.stft_stride):
                j = i + self.stft_frames
                slice_info = (spectrogram[0], i, j)
                metadata.append(slice_info)
        return metadata

    def get_full_vocal_spectrogram(self, spectrogram_id: int) -> np.ndarray:
        spectrogram_file_name = f"spec{spectrogram_id:06d}.npy"
        spectrogram_array = np.load(os.path.join(self.vox_path, spectrogram_file_name), mmap_mode='r')

        return spectrogram_array

    def get_full_mix_spectrogram(self, spectrogram_id: int) -> np.ndarray:
        spectrogram_file_name = f"spec{spectrogram_id:06d}.npy"
        spectrogram_array = np.load(os.path.join(self.mix_path, spectrogram_file_name), mmap_mode='r')

        return spectrogram_array

    def get_all_spectrogram_slices(self, spectrogram_id: int):
        spectrogram_id = f"spec{spectrogram_id:06d}"
        all_slice_info = self.metadata
        relevant_slice_only = [s for s in all_slice_info if s[0] == spectrogram_id]

        return relevant_slice_only

    def __getitem__(self, index):
        slice_info = self.metadata[index]
        fname = slice_info[0] + ".npy"
        i = slice_info[1]
        j = slice_info[2]
        x = np.load(os.path.join(self.mix_path, fname), mmap_mode='r')[:, :, i:j]
        y = np.load(os.path.join(self.vox_path, fname), mmap_mode='r')[:, i + self.offset]

        return x, y

    def __len__(self):
        return len(self.metadata)


class MyCollator(object):
    def __init__(self, mask_threshold):
        self.mask_threshold = mask_threshold

    def __call__(self, batch):
        x = [it[0] for it in batch]
        x = np.stack(x).astype(np.float32)
        x = torch.FloatTensor(x)
        y = [it[1] for it in batch]
        y = np.stack(y)
        if self.mask_threshold is not None:
            y = y > self.mask_threshold

        y = torch.FloatTensor(y.astype(np.float32))
        return x, y
