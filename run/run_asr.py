import hydra
import torch
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import ModelSummary
from asr_project.model.lightning_model import ASRModelLightening
from asr_project.data.lightning_data_module import ASRDataModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from omegaconf import OmegaConf, DictConfig


@hydra.main(config_path="../asr_project/config", config_name="config", version_base=None)
def asr_pipe(config: DictConfig):
    model = ASRModelLightening(config)
    data_module = ASRDataModule(config)
    # wandb.login(key = '599e3f8046b44d0d6d7fe5168ad43a34d81a9b20')
    logger = WandbLogger(name='asr_project', save_dir=config.model_dir, entity='huji_ayanas')
    logger.experiment.config.update(OmegaConf.to_container(config, resolve=True))
    OmegaConf.save(config, f"{logger.save_dir}/hydra_config.yaml")

    checkpoint_callback = ModelCheckpoint(monitor='val/wer',
                                          save_top_k=1, every_n_epochs=1)
    callbacks = [ModelSummary(max_depth=3), checkpoint_callback]
    trainer = Trainer(logger=logger,
                      callbacks=callbacks,
                      **config.trainer,
                      )
    trainer.fit(model=model, ckpt_path="last", datamodule=data_module)


if __name__ == '__main__':
    asr_pipe()