import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from lightning_callbacks import LogAll
from musdb_data_module import MUSDBDataModule
from vocal_mask_model import LitVocalMask, VocalMask


def train():
    # Setup data module
    torch.set_float32_matmul_precision('high')
    data_module = MUSDBDataModule()

    # Define the models
    hp = data_module.hparams
    fft_bins = hp.fft_size // 2 + 1
    input_dims = (fft_bins, hp.stft_frames)
    output_dims = fft_bins
    torch_model = VocalMask(input_dims=input_dims, output_dims=output_dims)
    lit_model = LitVocalMask(vocal_mask_model=torch_model)

    # define logger and callbacks
    logger = TensorBoardLogger(save_dir='..', default_hp_metric=False)
    checkpoint_callback = ModelCheckpoint(
        save_top_k=-1,
        dirpath='../checkpoints',
        save_on_train_epoch_end=True)

    # Define the trainer
    log_every_n_steps = 100
    check_val_every_n_epoch = 5

    trainer = Trainer(
        check_val_every_n_epoch=check_val_every_n_epoch,
        num_sanity_val_steps=0,
        accelerator='gpu',
        log_every_n_steps=log_every_n_steps,
        logger=logger,
        callbacks=[checkpoint_callback, LogAll()],
        max_epochs=100,
    )

    trainer.fit(model=lit_model, datamodule=data_module)
    trainer.test(lit_model, data_module)


if __name__ == '__main__':
    train()
