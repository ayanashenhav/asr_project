import hydra
import torch
from asr_project.model.lightning_model import ASRModelLightening
from asr_project.data.lightning_data_module import ASRDataModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from omegaconf import OmegaConf


if __name__ == '__main__':
    hydra.initialize("../asr_project/config")
    config = hydra.compose("config", overrides=["model_dir=/home/vpnuser/cs_huji/speech/asr_project/runs"])

    model = ASRModelLightening(config)
    data_module = ASRDataModule(config.data)

    logger = WandbLogger(name='asr_project', save_dir=config.model_dir, entity='huji_ayanas')
    logger.experiment.config.update(OmegaConf.to_container(config, resolve=True))

    # ask Amitai if we need this
    # checkpoint_callback = [ModelCheckpoint(monitor='val/loss', dirpath=config.model_dir, save_last=True,
    #                                        save_top_k=-1, every_n_epochs=10),
    #                        ModelCheckpoint(monitor='val/accuracy', mode='max', dirpath=config.model_dir, save_last=True,
    #                                        save_top_k=-1, every_n_epochs=10),
    #                        pl.callbacks.ModelSummary(max_depth=3)]

    checkpoint_callback = []

    trainer = Trainer(logger=logger,
                      callbacks=checkpoint_callback,
                      **config['trainer'],
                      )
    trainer.fit(model=model, ckpt_path="last", datamodule=data_module)
