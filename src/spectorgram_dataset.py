import os
import numpy as np
import torch

from torch.utils.data import Dataset
from tqdm import tqdm


class SpectrogramDataset(Dataset):
    def __init__(self, data_path, spec_info, stft_frames, stft_stride):
        self.data_path = data_path
        self.mix_path = os.path.join(data_path, "spec_mix")
        self.vox_path = os.path.join(data_path, "spec_vox")
        self.stft_frames = stft_frames
        self.stft_stride = stft_stride
        self.offset = stft_frames // 2
        self.metadata = self.get_slices(spec_info)

    def get_slices(self, spec_info):
        metadata = []
        print("Preparing dataset")
        for spec in tqdm(spec_info):
            size = spec[1] - self.stft_frames
            for i in range(0, size, self.stft_stride):
                j = i + self.stft_frames
                slice_info = (spec[0], i, j)
                metadata.append(slice_info)
        return metadata

    def get_full_vocal_spectrogram(self, spec_id: int) -> np.ndarray:
        spec_file_name = f"spec{spec_id:06d}.npy"
        spec_array = np.load(os.path.join(self.vox_path, spec_file_name), mmap_mode='r')

        return spec_array

    def get_full_mix_spectrogram(self, spec_id: int) -> np.ndarray:
        spec_file_name = f"spec{spec_id:06d}.npy"
        spec_array = np.load(os.path.join(self.mix_path, spec_file_name), mmap_mode='r')

        return spec_array

    def get_all_spectrogram_slices(self, spec_id: int):
        # if spec_id=16 then return 'spec000016'
        spec_id = f"spec{spec_id:06d}"
        all_slice_info = self.metadata

        # If spec_id=16 return [('spec000016', 0, 25), ('spec000016', 1, 26), ('spec000016', 2, 27), ('spec000016', 3, 28)...]
        relevant_slice_only = [s for s in all_slice_info if s[0] == spec_id]

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


def basic_collate(batch, mask_threshold):
    x = [it[0] for it in batch]
    x = np.stack(x).astype(np.float32)
    x = torch.FloatTensor(x)
    y = [it[1] for it in batch]
    y = np.stack(y)
    if mask_threshold is not None:
        y = y > mask_threshold

    y = torch.FloatTensor(y.astype(np.float32))
    return x, y
