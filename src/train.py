from pytorch_lightning import Trainer

from musdb_data_module import MUSDBDataModule
from vocal_mask_model import LitVocalMask, VocalMask


# Setup data module
data_module = MUSDBDataModule()

# Define the models
hp = data_module.hparams
fft_bins = hp.fft_size // 2 + 1
input_dims = (fft_bins, hp.stft_frames)
output_dims = fft_bins
torch_model = VocalMask(input_dims=input_dims, output_dims=output_dims)
lit_model = LitVocalMask(vocal_mask_model=torch_model)

# Define the trainer
trainer = Trainer()

# Train
trainer.fit(
    model=lit_model,
    datamodule=data_module
)



