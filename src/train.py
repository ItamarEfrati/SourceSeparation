from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from lightning_callbacks import LogVocalSeparationGridCallback
from musdb_data_module import MUSDBDataModule
from vocal_mask_model import LitVocalMask, VocalMask


def main():
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
    logger = TensorBoardLogger(save_dir=r"C:\git\SourceSeparation\src\lightning_logs")

    log_every_n_steps = 1000
    trainer = Trainer(
        log_every_n_steps=log_every_n_steps,
        logger=logger,
        callbacks=[LogVocalSeparationGridCallback(data_module, 16, every_n_steps=log_every_n_steps)]
    )

    # Train
    trainer.fit(
        model=lit_model,
        datamodule=data_module
    )


if __name__ == '__main__':
    main()
